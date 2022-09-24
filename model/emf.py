import imp
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from dataloader.loader import Lang
from util import config
import math
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(LightningModule):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer(
            'pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i)
                                  for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Gennerator(LightningModule):
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        # self.log_softmax = nn.LogSoftmax(dim=1)
        # self.loss = nn.NLLLoss()

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x


class ScaledDotProductAttention(LightningModule):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, scale_dim, attn_dropout=0.1):
        super().__init__()
        self.scale_dim = scale_dim
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        # attn = torch.matmul(q / self., k.transpose(2, 3))
        #query: [batch_size, n_heads, len_q, key_dim]
        #key: [batch_size, n_heads, len_k, key_dim]
        attn = (q/self.scale_dim)@k.transpose(2, 3)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        # output = torch.matmul(attn, v)
        #value: [batch_size, n_heads, len_v, val_dim]
        # asset len_key==len_value
        #output: [batch_size, n_heads, len_q, val_dim]
        output = attn @ v

        return output, attn


class MultiHeadAttention(LightningModule):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, hid_dim, dropout=0.1):
        '''
        :param n_head: number of heads
        :param hid_dim: hidden dimension
        '''
        super().__init__()

        self.n_head = n_head
        self.head_dim=hid_dim//n_head
        # hid_dim could not equal to n_head * key_dim
        self.fc_q = nn.Linear(hid_dim, n_head * self.head_dim, bias=False)
        self.fc_k = nn.Linear(hid_dim, n_head * self.head_dim, bias=False)
        self.fc_v = nn.Linear(hid_dim, n_head * self.head_dim, bias=False)
        self.fc_o = nn.Linear(hid_dim, hid_dim, bias=False)

        self.attention = ScaledDotProductAttention(scale_dim=hid_dim ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)

    def forward(self, query, key, value, mask=None):

        # batch first ensured
        batch_size, query_len, key_len, val_len = query.size(
            0), query.size(1), key.size(1), value.size(1)

        # residual = query

        # Pass through the pre-attention projection: batch_size x len_q x n_head x d_k
        # Separate different heads: batch_size x n_head x len_q x d_k
        Q = self.fc_q(query).view(batch_size, query_len,
                                  self.n_head, self.head_dim).transpose(1, 2)
        K = self.fc_k(key).view(batch_size, key_len,
                                self.n_head,self.head_dim).transpose(1, 2)
        V = self.fc_v(value).view(batch_size, val_len,
                                  self.n_head,self.head_dim).transpose(1, 2)


        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        out, attn = self.attention(Q, K, V, mask=mask)
        #attn = torch.softmax(attn, dim=-1)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        out = out.transpose(1, 2).contiguous().view(batch_size, query_len, -1)
        out = self.dropout(self.fc_o(out))
        # residual connection
        out += query
        out = self.layer_norm(out)
        return out, attn


class MultiHeadAttentionMutual(LightningModule):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head: int,
                 hid_dim: int,
                 key_dim: int,
                 val_dim: int,
                 dropout=0.1,
                 query_hid_dim: int = None,
                 key_hid_dim: int = None,
                 val_hid_dim: int = None
                 ):
        '''
        :param n_head: number of heads
        :param query_dim: dimension of query
        :param key_dim: dimension of key
        :param val_dim: dimension of value
        '''
        super().__init__()
        #if hid_dim is not None,then query,key,value hid_dim is hid_dim
        query_hid_dim = hid_dim if query_hid_dim is None else query_hid_dim
        key_hid_dim = hid_dim if key_hid_dim is None else key_hid_dim
        val_hid_dim = hid_dim if val_hid_dim is None else val_hid_dim
        
        self.n_head = n_head
        self.key_dim = key_dim
        self.val_dim = val_dim
        # assert hid_dim == n_head * key_dim
        # hid_dim could not equal to n_head * key_dim
        self.fc_q = nn.Linear(query_hid_dim, n_head * key_dim, bias=False)
        self.fc_k = nn.Linear(key_hid_dim, n_head * key_dim, bias=False)
        self.fc_v = nn.Linear(val_hid_dim, n_head * val_dim, bias=False)
        self.fc_o = nn.Linear(n_head * val_dim, query_hid_dim, bias=False)

        self.attention = ScaledDotProductAttention(scale_dim=key_dim ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_head*val_dim, eps=1e-6)

    def forward(self, query, key, value, mask=None):

        key_dim, val_dim, n_head = self.key_dim, self.val_dim, self.n_head
        # batch first ensured
        batch_size, query_len, key_len, val_len = query.size(
            0), query.size(1), key.size(1), value.size(1)

        # residual = query

        # Pass through the pre-attention projection: batch_size x len_q x n_head x d_k
        # Separate different heads: batch_size x n_head x len_q x d_k
        Q = self.fc_q(query).view(batch_size, query_len,
                                  n_head, key_dim).transpose(1, 2)
        K = self.fc_k(key).view(batch_size, key_len,
                                n_head, key_dim).transpose(1, 2)
        V = self.fc_v(value).view(batch_size, val_len,
                                  n_head, val_dim).transpose(1, 2)

        # Transpose for attention dot product: b x n x lq x dv
        # q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        out, attn = self.attention(Q, K, V, mask=mask)
        #attn = torch.softmax(attn, dim=-1)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        out = out.transpose(1, 2).contiguous().view(batch_size, query_len, -1)
        out = self.dropout(self.fc_o(out))
        # residual connection
        out += query
        out = self.layer_norm(out)
        return out, attn


class PositionwiseFeedForward(LightningModule):
    ''' A two-feed-forward-layer module '''

    def __init__(self, hid_dim, pf_dim, dropout=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)  # position-wise
        self.fc_2 = nn.Linear(hid_dim, pf_dim)  # position-wise
        self.layer_norm = nn.LayerNorm(pf_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        X = self.fc_2(F.relu(self.fc_1(x)))
        X = self.dropout(X)

        # residual connection
        X += x
        X = self.layer_norm(X)
        return X


class EncoderLayer(LightningModule):
    ''' Compose with two layers '''

    def __init__(self, hid_dim, pf_dim, n_head, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, hid_dim, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            hid_dim, pf_dim, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):

        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, input_dim, hid_dim, n_layers, n_head,
            pf_dim, dropout=0.1, max_len=1000, scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Embedding(
            input_dim, hid_dim, padding_idx=config.PAD_idx)
        self.position_enc = PositionalEncoding(
            hid_dim, n_position=max_len)
        self.dropout = nn.Dropout(dropout)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(hid_dim, pf_dim, n_head, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.scale_emb = scale_emb
        self.hid_dim = hid_dim

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.hid_dim ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.enc_layers:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class DecoderLayer(LightningModule):
    ''' Compose with three layers '''

    def __init__(self, hid_dim, pf_dim, n_head, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, hid_dim, dropout=dropout)
        self.enc_attn = MultiHeadAttention(
            n_head, hid_dim, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            hid_dim, pf_dim, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        # using broadcast mechanism
        #self_attn_mask: [batch_size, 1, len_q, len_q]
        #dec_enc_attn_mask: [batch_size, 1, len_q, len_k]
        #residual in slf_attn
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)

        #residual in enc_attn
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)

        #residual in pos_ffn
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
        self, output_dim, hid_dim, n_layers, n_head,
            pf_dim, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(
            output_dim, hid_dim, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(
            hid_dim, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.dec_layers = nn.ModuleList([
            DecoderLayer(hid_dim, pf_dim, n_head, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.scale_emb = scale_emb
        self.hid_dim = hid_dim

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.hid_dim ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.dec_layers:
            #dec_output: [batch_size, trg_seq_len-1, hid_dim]
            #enc_output: [batch_size, src_seq_len, n_head*val_dim]
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class EMF(LightningModule):
    def __init__(self, vocab: Lang):
        super(EMF, self).__init__()
        self.vocab = vocab
        # self.encoder = Encoder()
        self.encoder = Encoder(
            input_dim=vocab.n_words
        )

    def create_mask(self, src, tgt, batch_first: bool = True):
        if not batch_first:
            src = src.transpose(0, 1)
            tgt = tgt.transpose(0, 1)
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_seq_len).to(self.device)
        src_mask = torch.zeros((src_seq_len, src_seq_len),
                               device=self.device).type(torch.bool)

        src_padding_mask = (src == config.PAD_idx)
        tgt_padding_mask = (tgt == config.PAD_idx)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, input_batch, target_batch, target_program):
        # divide data
        ref_batch = target_batch[:, 1:]
        target_batch = target_batch[:, :-1]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(
            input_batch, target_batch)
        # encode
        context = self.context_embedding(input_batch)
        # context = self.position_embedding(context)
        context = self.dropout(context)
        context = self.transformer_encoder(context, src_mask, src_padding_mask)
        # decode
        target = self.context_embedding(target_batch)
        # target = self.position_embedding(target)
        target = self.dropout(target)
        output = self.transformer_decoder(target, context, tgt_mask)
        # output = self.dropout(output)
        # output = self.linear(output)
        output = self.generator(output)
        loss = F.nll_loss(output.contiguous().view(-1, output.size(-1)),
                          ref_batch.contiguous().view(-1),
                          ignore_index=config.PAD_idx)
        return loss

    def init_weights(self) -> None:
        pass

    def get_input_batch(self, batch):
        input_batch = batch['input_batch']
        input_mask = batch['input_mask']
        target_batch = batch['target_batch']
        target_program = batch['target_program']
        return input_batch, input_mask, target_batch, target_program

    def training_step(self, batch, batch_idx):
        input_batch, input_mask, target_batch, target_program = self.get_input_batch(
            batch)
        loss = self(input_batch, target_batch, target_program)
        self.log('train_loss', loss)
        self.log('train_ppl', math.exp(loss.item()))

        return loss

    def validation_step(self, batch, batch_idx):
        input_batch, input_mask, target_batch, target_program = self.get_input_batch(
            batch)
        loss = self(input_batch, target_batch, target_program)
        self.log('valid_loss', loss)
        self.log('valid_ppl', math.exp(loss.item()))
        return loss

    def test_step(self, batch, batch_idx):

        input_batch, input_mask, target_batch, target_program = self.get_input_batch(
            batch)
        trg_tensors = self.decoder_greedy(input_batch, max_length=1000)
        pass

    def decoder_greedy(self, input_batch, max_length):
        context = self.context_embedding(input_batch)
        trg_tensor = torch.zeros(
            (input_batch.size(0), max_length)).long().to(self.device)
        trg_tensor[:, 0] = config.SOS_idx
        for i in range(1, max_length):
            src_mask, trg_mask, src_padding_mask, tgt_padding_mask = self.create_mask(
                input_batch, trg_tensor)
            output = self.transformer_decoder(
                self.context_embedding(trg_tensor), context, trg_mask)
            output = self.generator(output)
            _, next_word = torch.max(output[:, -1], dim=1)
            output = output.argmax(dim=-1)
            trg_tensor[:, i] = output[:, i]
        return trg_tensor

    def configure_optimizers(self):
        return
        return torch.optim.Adam(self.parameters(), lr=1e-3)
