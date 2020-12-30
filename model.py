from math import sqrt, ceil
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
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

        self.range_projection = LinearNorm(2*hparams.range_rnn_dim, 1, bias=False)

    def positional_embedding(self, c, dim, T):
        positional_embedding = torch.zeros(T, dim, device=c.device)
        i = torch.arange(1, dim+1, 2, device=c.device).float()
        pos = torch.arange(1, T+1, device=c.device).unsqueeze(1).expand(-1, dim // 2)


        divisor = torch.pow(self.timestep_denominator, i / float(dim))

        positional_embedding[:, 0::2] = torch.sin(pos.float() / divisor)
        positional_embedding[:, 1::2] = torch.cos(pos.float() / divisor)

        return positional_embedding

    def gaussian_upsampling(self, H, d, c, std_dev, T):
        # H = (seq_len, emb_size)
        # d = (seq_len)
        # std_dev = (seq_len) == sigma == σ


        # T = c_N, N = seq_len
        N = H.size(0)
        # U = (time_steps, emb_size)
        t = torch.arange(1, T+1, device=c.device).unsqueeze(0).transpose(0, 1)

        c = c - (d / 2)
        dividend = torch.exp(-std_dev**-2. * (t-c)**2.).t()
        divisor = torch.sum(torch.exp(-std_dev**-2. * (t - c)**2.), 1)
        w = dividend / divisor
        U = torch.matmul(w.t(), H)

        return U

    def forward(self, h, d, max_T):
        # Convert frame size in seconds to integer frames
        d = torch.round(d * self.frame_size).long()
        c = torch.cumsum(d, -1)
        # U = (batch_size, time_steps, emb_size)
        # Add 1 to last dimension of U for positional embedding
        U = torch.empty(h.size(0), max_T, self.encoder_embedding_dim + self.positional_embedding_dim, device=d.device)

        self.duration_lstm.flatten_parameters()
        duration_h, _ = self.duration_lstm(h)
        pred_duration_output = self.duration_projection(duration_h)
        # (batch_size, seq_len, 1) -> (batch_size, seq_len)
        pred_duration_output = pred_duration_output.squeeze()

        self.range_lstm.flatten_parameters()
        range_h, _ = self.range_lstm(torch.cat((h, d.unsqueeze(-1)), -1))
        range_output = F.softplus(self.range_projection(range_h))
        range_output = range_output.squeeze()



        for i in range(len(h)):
            gaussian = self.gaussian_upsampling(h[i], d[i], c[i], range_output[i], max_T)
            positional = self.positional_embedding(c[i], self.positional_embedding_dim, max_T)
            
            assert len(gaussian) == len(positional), "Gaussian output and positional output should be same size"

            for j in range(len(gaussian)):
                g, p = gaussian[j], positional[j]
                length = g.size(0)

                U[i, :length, ...] = torch.cat((g, p), -1)

        return U, pred_duration_output


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
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
            outputs = zoneout(outputs[0], outputs, p=0.1)

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

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.decoder_rnn1 = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim + hparams.positional_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.decoder_rnn2 = nn.LSTMCell(
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

    def decode(self, decoder_input, duration_input):
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
        cell_input = torch.cat((decoder_input, duration_input), -1)

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
            (self.decoder_hidden2, duration_input), dim=1)
        decoder_output = self.linear_projection(hidden_duration_context)

        return decoder_output

    def forward(self, duration_outputs, decoder_inputs, memory_lengths):
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


        decoder_input = self.get_go_frame(duration_outputs.size(0), decoder_inputs.device).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        mask=~get_mask_from_lengths(memory_lengths)

        self.decoder_hidden1 = torch.zeros(B, self.decoder_rnn_dim, device=decoder_inputs.device)
        self.decoder_cell1 = torch.zeros(B, self.decoder_rnn_dim, device=decoder_inputs.device)
        self.decoder_hidden2 = torch.zeros(B, self.decoder_rnn_dim, device=decoder_inputs.device)
        self.decoder_cell2 = torch.zeros(B, self.decoder_rnn_dim, device=decoder_inputs.device)

        self.mask = mask

        duration_outputs = duration_outputs.transpose(0, 1)

        mel_outputs = []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            mel_outputs_len = len(mel_outputs)
            decoder_input = decoder_inputs[mel_outputs_len]
            duration_input = duration_outputs[mel_outputs_len]
            mel_output = self.decode(
                decoder_input, duration_input)
            mel_outputs += [mel_output.squeeze(1)]

        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        memory == duration_outputs
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


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
        self.duration_aligner = Duration(hparams)
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

        MAX_T = mels.size(-1)
        
        duration_outputs, predicted_durations = self.duration_aligner(encoder_outputs, durations, MAX_T)

        mel_outputs = self.decoder(duration_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, predicted_durations],
            output_lengths)

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
