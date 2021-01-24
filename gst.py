import math
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class TPSE(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        
        self.rnn = nn.GRU(hparams.encoder_output_dim, hparams.tpse_rnn_size, batch_first=True)
        self.fc1 = nn.Linear(hparams.tpse_rnn_size, hparams.gst_embedding_dim)
        
    def forward(self, x):
        self.rnn.flatten_parameters()
        _, x = self.rnn(x)
        x = x.squeeze(0)
        x = self.fc1(x)
        x = x.tanh()
        x = x.unsqueeze(1)

        return x
        

class GST(nn.Module):

    def __init__(self, hparams):

        super().__init__()
        self.encoder = ReferenceEncoder(hparams)
        self.styletoken = StyleTokenLayer(hparams)

    def forward(self, x):
        x = self.encoder(x)
        x = self.styletoken(x)

        return x


class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, hparams):
        super().__init__()
        self.n_mels = hparams.n_mel_channels
        output_channel_count = len(hparams.gst_output_channels) # output_channel_count = K
        filters = [1] + hparams.gst_output_channels
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=hparams.gst_kernel_size,
                           stride=hparams.gst_stride,
                           padding=hparams.gst_padding) for i in range(output_channel_count)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=hparams.gst_output_channels[i]) for i in range(output_channel_count)])

        channel_2d_size = self.calculate_channels(self.n_mels, 3, 2, 1, output_channel_count)
        self.gru = nn.GRU(input_size=hparams.gst_output_channels[-1] * channel_2d_size,
                          hidden_size=hparams.gst_embedding_dim // 2,
                          batch_first=True)

    def forward(self, x):
        B = x.size(0)
        # x = x.view(B, 1, -1, self.n_mels)  # [B, 1, Ty, n_mels]
        x = x.unsqueeze(1) # [B, 1, N, T]
        x = x.transpose(2, 3) # [B, 1, T, N]
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)  # [B, 128, T//2^K, N//2^K]

        x = x.transpose(1, 2)  # [B, T//2^K, 128, N//2^K]
        B, T = x.size(0), x.size(1)
        x = x.reshape(B, T, -1)  # [B, T//2^K, 128*N//2^K]

        self.gru.flatten_parameters()
        _, x = self.gru(x)  # out --- [1, N, E//2]

        return x.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class StyleTokenLayer(nn.Module):
    '''
    inputs --- [N, E//2]
    '''

    def __init__(self, hparams):

        super().__init__()
        d_q = hparams.gst_embedding_dim // 2
        d_k = hparams.gst_embedding_dim // hparams.gst_heads
        dim_input = hparams.gst_embedding_dim // 2
        dim_key_query_all = hparams.gst_embedding_dim // 8
        self.attention = CollaborativeAttention(dim_input=dim_input, 
                                                dim_value_all=hparams.gst_embedding_dim, 
                                                dim_key_query_all=dim_key_query_all,
                                                num_attention_heads=hparams.gst_heads)

    def forward(self, x):
        N = x.size(0)
        x = x.unsqueeze(1)  # [N, 1, E//2]
        x = x.tanh()
        x = self.attention(x)

        return x

class MixingMatrixInit(Enum):
    CONCATENATE = 1
    ALL_ONES = 2
    UNIFORM = 3


class CollaborativeAttention(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_value_all: int,
        dim_key_query_all: int,
        num_attention_heads: int,
        mixing_initialization: MixingMatrixInit = MixingMatrixInit.UNIFORM,
    ):
        super().__init__()

        if dim_value_all % num_attention_heads != 0:
            raise ValueError(
                "Value dimension ({}) should be divisible by number of heads ({})".format(
                    dim_value_all, num_attention_heads
                )
            )

        # save args
        self.dim_input = dim_input
        self.dim_value_all = dim_value_all
        self.dim_key_query_all = dim_key_query_all
        self.num_attention_heads = num_attention_heads
        self.mixing_initialization = mixing_initialization

        self.dim_value_per_head = dim_value_all // num_attention_heads
        self.attention_head_size = (
            dim_key_query_all / num_attention_heads
        )  # does not have to be integer

        # intialize parameters
        self.query = nn.Linear(dim_input, dim_key_query_all, bias=False)
        self.key = nn.Linear(dim_input, dim_key_query_all, bias=False)
        self.value = nn.Linear(dim_input, dim_value_all)

        self.mixing = self.init_mixing_matrix()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        from_sequence = hidden_states
        to_sequence = hidden_states
        if encoder_hidden_states is not None:
            to_sequence = encoder_hidden_states
            attention_mask = encoder_attention_mask

        query_layer = self.query(from_sequence)
        key_layer = self.key(to_sequence)

        # point wise multiplication of the mixing coefficient per head with the shared query projection
        # (batch, from_seq, dim) x (head, dim) -> (batch, head, from_seq, dim)
        mixed_query = query_layer[..., None, :, :] * self.mixing[..., :, None, :]

        # broadcast the shared key for all the heads
        # (batch, 1, to_seq, dim)
        mixed_key = key_layer[..., None, :, :]

        # (batch, head, from_seq, to_seq)
        attention_scores = torch.matmul(mixed_query, mixed_key.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        value_layer = self.value(to_sequence)
        value_layer = self.transpose_for_scores(value_layer)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.dim_value_all,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def init_mixing_matrix(self, scale=0.2):
        mixing = torch.zeros(self.num_attention_heads, self.dim_key_query_all)

        if self.mixing_initialization is MixingMatrixInit.CONCATENATE:
            # last head will be smaller if not equally divisible
            dim_head = int(math.ceil(self.dim_key_query_all / self.num_attention_heads))
            for i in range(self.num_attention_heads):
                mixing[i, i * dim_head : (i + 1) * dim_head] = 1.0

        elif self.mixing_initialization is MixingMatrixInit.ALL_ONES:
            mixing.one_()
        elif self.mixing_initialization is MixingMatrixInit.UNIFORM:
            mixing.normal_(std=scale)
        else:
            raise ValueError(
                "Unknown mixing matrix initialization: {}".format(
                    self.mixing_initialization
                )
            )

        return nn.Parameter(mixing)


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, x, key):
        x = self.W_query(x)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        x = torch.stack(torch.split(x, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        x = torch.matmul(x, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        x = x / (self.key_dim ** 0.5)
        x = F.softmax(x, dim=3)

        # out = score * V
        x = torch.matmul(x, values)  # [h, N, T_q, num_units/h]
        x = torch.cat(torch.split(x, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return x
