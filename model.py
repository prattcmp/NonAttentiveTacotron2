from math import sqrt, ceil, log
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm, LSTMCellNorm
from utils import to_gpu, get_mask_from_lengths


def zoneout(prev, current, p=0.1):
    mask = torch.empty_like(prev).bernoulli_(p)
    return mask * prev + (1 - mask) * current

class Duration(nn.Module):
    def __init__(self, hparams):
        super(Duration, self).__init__()
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.positional_embedding_dim = hparams.positional_embedding_dim
        self.timestep_denominator = hparams.timestep_denominator
        # Frame size in ms
        self.frame_size = hparams.sampling_rate / hparams.hop_length
        
        # Duration predictor
        self.duration_lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            hparams.duration_rnn_dim, 2,
                            batch_first=True, bidirectional=True)

        # Range parameter predictor
        # Add 1 because we concatenate durations
        self.range_lstm = nn.LSTM(hparams.encoder_embedding_dim + 1,
                            hparams.range_rnn_dim, 2,
                            batch_first=True, bidirectional=True)

        self.duration_projection = LinearNorm(2* hparams.duration_rnn_dim, 1, bias=False)
        #self.duration_projection = torch.nn.Linear(2*hparams.duration_rnn_dim, 1)

        self.range_projection = LinearNorm(2*hparams.range_rnn_dim, 1, bias=False)
        #self.range_projection = torch.nn.Linear(2*hparams.range_rnn_dim, 1)

        self.score_mask_value = -float("inf")

    def positional_embedding(self, c, dim, t):
        positional_embedding = torch.zeros(c.size(0), dim, device=c.device)
        i = torch.arange(0, dim, 2, device=c.device).float()
        #pos = torch.arange(0, T, device=c.device).unsqueeze(1).expand(-1, dim // 2)
        pos = t


        divisor = torch.exp(i * -(log(self.timestep_denominator) / float(dim)))

        positional_embedding[:, 0::2] = torch.sin(float(pos) * divisor)
        positional_embedding[:, 1::2] = torch.cos(float(pos) * divisor)

        return positional_embedding

    def gaussian_upsampling(self, H, d, c, std_dev, t):
        # H = (seq_len, emb_size)
        # d = (seq_len)
        # std_dev = (seq_len) == sigma == σ


        # T = c_N, N = seq_len
        #N = H.size(1)
        # U = (time_steps, emb_size)
        #t = torch.arange(1, t+1, device=c.device).unsqueeze(0).transpose(0, 1)
        #t = t.unsqueeze(0).expand(d.size(0), -1, -1)


        c = c - (d / 2)
        #t_c = torch.sub(t, c) ** 2.
        #pow_std_dev = (-1 / (std_dev ** 2.)).unsqueeze(1).expand_as(t_c)
        #print(pow_std_dev)
        #power = (t_c * pow_std_dev)
        power = (torch.abs(t-c)**2. / -1 * (std_dev**2.))

        if self.mask is not None:
            power.data.masked_fill_(self.mask, self.score_mask_value)
        w = F.softmax(power, -1).unsqueeze(1)
        U = torch.matmul(w, H)

        return U, w

    def forward(self, h, d, mask, input_lengths):
        self.mask=mask

        range_in = torch.cat((h, d.unsqueeze(-1)), -1)
        input_lengths = input_lengths.cpu().numpy()
        range_in = nn.utils.rnn.pack_padded_sequence(
            range_in, input_lengths, batch_first=True)
        h1 = nn.utils.rnn.pack_padded_sequence(
            h, input_lengths, batch_first=True)

        self.duration_lstm.flatten_parameters()
        duration_h, _ = self.duration_lstm(h1)
        duration_h, _ = nn.utils.rnn.pad_packed_sequence(
            duration_h, batch_first=True)
        pred_duration_output = self.duration_projection(duration_h)
        # (batch_size, seq_len, 1) -> (batch_size, seq_len)
        pred_duration_output = pred_duration_output.squeeze()


        self.range_lstm.flatten_parameters()

        range_h, _ = self.range_lstm(range_in)
        range_h, _ = nn.utils.rnn.pad_packed_sequence(
            range_h, batch_first=True)
        range_proj = self.range_projection(range_h)
        range_output = F.softplus(range_proj)
        range_output = range_output.squeeze()

        return range_output, pred_duration_output

    def inference(self, h):
        self.duration_lstm.flatten_parameters()
        duration_h, _ = self.duration_lstm(h)
        d = self.duration_projection(duration_h)
        # (batch_size, seq_len, 1) -> (batch_size, seq_len)
        d = d.squeeze()

        self.range_lstm.flatten_parameters()
        range_h, _ = self.range_lstm(torch.cat((h, d.unsqueeze(-1)), -1))
        range_output = F.softplus(self.range_projection(range_h))
        range_output = range_output.squeeze()

        # Convert frame size in seconds to integer frames
        d = torch.round(d * self.frame_size).long()
        # Find the longest sequence (in frames)
        c = torch.cumsum(d, -1)
        max_T = ceil(torch.max(c))

        # U = (batch_size, time_steps, emb_size)
        U = torch.empty(h.size(0), max_T, self.encoder_embedding_dim + self.positional_embedding_dim, device=d.device)
        weights = torch.empty(h.size(0), max_T, self.encoder_embedding_dim, device=d.device)

        for i in range(len(h)):
            gaussian, weight = self.gaussian_upsampling(h[i], d[i], c[i], range_output[i], max_T)
            positional = self.positional_embedding(c[i], self.positional_embedding_dim, max_T)
            
            assert len(gaussian) == len(positional), "Gaussian output and positional output should be same size"

            for j in range(len(gaussian)):
                g, w, p = gaussian[j], weight[j], positional[j]
                length = g.size(0)

                U[i, :length, ...] = torch.cat((g, p), -1)
                weights[i, :length, ...] = w

        return U, weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        i = 0
        for linear in self.layers:
            i += 1
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dims[0],
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dims[0]))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dims[i-1],
                             hparams.postnet_embedding_dims[i],
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dims[i]))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dims[hparams.postnet_n_convolutions-2], hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        if self.training:
            outputs = F.dropout(outputs, 0.1, self.training)
            #outputs = zoneout(outputs[0], outputs, p=0.1)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.duration_aligner = Duration(hparams)

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.decoder_rnn1 = LSTMCellNorm(
            hparams.prenet_dim + hparams.encoder_embedding_dim + hparams.positional_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.decoder_rnn2 = LSTMCellNorm(
            hparams.decoder_rnn_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim + hparams.positional_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

    def get_go_frame(self, B, device):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        B: batch size

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        return torch.zeros(B, self.n_mel_channels * self.n_frames_per_step, device=device)

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def decode(self, decoder_input, text_lengths, t, range_outputs, d, c):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output
        duration_input: duration aligner output for current frame

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """

        
        self.gaussian_upsampling, self.weights = self.duration_aligner.gaussian_upsampling(self.encoder_outputs, d, c, range_outputs, t)
        self.positional_embedding = self.duration_aligner.positional_embedding(c, self.duration_aligner.positional_embedding_dim, t)
        
        self.duration_output = torch.cat((self.gaussian_upsampling.squeeze(), self.positional_embedding), -1)

        cell_input = torch.cat((decoder_input, self.duration_output), -1)

        self.decoder_hidden1, self.decoder_cell1 = self.decoder_rnn1(
            cell_input, (self.decoder_hidden1, self.decoder_cell1))

        if self.training:
            self.decoder_hidden1 = zoneout(
                self.decoder_hidden1[0], self.decoder_hidden1, p=0.1)

        self.decoder_hidden2, self.decoder_cell2 = self.decoder_rnn2(
            self.decoder_hidden1, (self.decoder_hidden2, self.decoder_cell2))
        if self.training:
            self.decoder_hidden2 = zoneout(
                self.decoder_hidden2[0], self.decoder_hidden2, p=0.1)

        hidden_duration_context = torch.cat(
            (self.decoder_hidden2, self.duration_output), dim=1)
        decoder_output = self.linear_projection(hidden_duration_context)

        return decoder_output, self.weights

    def forward(self, encoder_outputs, durations, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        duration_outputs: Duration aligner outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        memory == duration_outputs
        """

        B = decoder_inputs.size(0)
        MAX_T = decoder_inputs.size(1)
        seq_len = durations.size(1)
        
        decoder_input = self.get_go_frame(B, decoder_inputs.device).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        mask=~get_mask_from_lengths(memory_lengths)

        self.gaussian_upsampling = torch.empty(B, self.duration_aligner.encoder_embedding_dim, device=decoder_inputs.device)
        self.positional_embedding = torch.empty(B, self.duration_aligner.positional_embedding_dim, device=decoder_inputs.device)
        self.weights = torch.empty(B, 1, seq_len, device=decoder_inputs.device)
        self.duration_output = torch.empty(B, self.encoder_embedding_dim + self.duration_aligner.positional_embedding_dim, device=decoder_inputs.device)

        self.decoder_hidden1 = torch.zeros(B, self.decoder_rnn_dim, device=decoder_inputs.device)
        self.decoder_cell1 = torch.zeros(B, self.decoder_rnn_dim, device=decoder_inputs.device)
        self.decoder_hidden2 = torch.zeros(B, self.decoder_rnn_dim, device=decoder_inputs.device)
        self.decoder_cell2 = torch.zeros(B, self.decoder_rnn_dim, device=decoder_inputs.device)

        self.mask = mask
        self.input_lengths = memory_lengths
        self.encoder_outputs = encoder_outputs
        self.durations = durations

        # Convert frame size in seconds to integer frames
        d = torch.round(durations * self.duration_aligner.frame_size).long()
        c = torch.cumsum(d, -1)


        range_outputs, predicted_durations = self.duration_aligner(encoder_outputs, d, self.mask, self.input_lengths)

        mel_outputs, alignments = [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            mel_outputs_len = len(mel_outputs)
            decoder_input = decoder_inputs[mel_outputs_len]

            mel_output, alignment = self.decode(
                decoder_input, memory_lengths, mel_outputs_len, range_outputs, d, c)
            mel_outputs += [mel_output.squeeze(1)]
            alignments += [alignment.squeeze(1)]

        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs)

        mel_outputs = mel_outputs.transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        alignments = torch.stack(alignments)

        return mel_outputs, predicted_durations, alignments

    def inference(self, duration_outputs):
        """ Decoder forward pass for training
        PARAMS
        ------
        duration_outputs: Duration aligner outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        memory == duration_outputs
        """

        decoder_input = self.get_go_frame(duration_outputs.size(0), decoder_inputs.device).unsqueeze(0)

        self.decoder_hidden1 = torch.zeros(B, self.decoder_rnn_dim, device=duration_outputs.device)
        self.decoder_cell1 = torch.zeros(B, self.decoder_rnn_dim, device=duration_outputs.device)
        self.decoder_hidden2 = torch.zeros(B, self.decoder_rnn_dim, device=duration_outputs.device)
        self.decoder_cell2 = torch.zeros(B, self.decoder_rnn_dim, device=duration_outputs.device)

        duration_outputs = duration_outputs.transpose(0, 1)

        mel_outputs = []
        while len(mel_outputs) < decoder_outputs.size(0):
            mel_outputs_len = len(mel_outputs)

            decoder_input = self.prenet(decoder_input)

            duration_input = duration_outputs[mel_outputs_len]

            mel_output = self.decode(decoder_input, duration_input)
            mel_outputs += [mel_output.squeeze(1)]

            decoder_input = mel_output

        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, duration_padded, \
            output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        duration_padded = to_gpu(duration_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, duration_padded, max_len, output_lengths),
            (mel_padded, duration_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)

        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, durations, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, predicted_durations, alignments = self.decoder(encoder_outputs, durations, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, predicted_durations, alignments],
            output_lengths)

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)

        duration_outputs, alignments = self.duration_aligner.inference(encoder_outputs)

        mel_outputs = self.decoder.inference(duration_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, alignments])

        return outputs
