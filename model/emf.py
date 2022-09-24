import imp
from turtle import forward
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


class LinearPosionEncoding(LightningModule):
    def __init__(self,max_len,hid_dim):
        super(LinearPosionEncoding,self).__init__()
        #sin cos position encoding
        # self.pe = nn.Embedding(max_len,hid_dim)
        # self.pe.weight.data = self._get_sinusoid_encoding_table(max_len,hid_dim)
        # self.pe.weight.requires_grad = False
        self.pos_emb = nn.Embedding(max_len,hid_dim)
    def forward(self,x):
        pos = torch.arange(0,x.size(1)).unsqueeze(0).repeat(x.size(0),1).to(x.device)
        return x + self.pos_emb(pos)
        # return self.pos_emb(x)
    



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
        self.head_dim = hid_dim//n_head
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
                                self.n_head, self.head_dim).transpose(1, 2)
        V = self.fc_v(value).view(batch_size, val_len,
                                  self.n_head, self.head_dim).transpose(1, 2)

        # if mask is not None:
        #     mask = mask.unsqueeze(1)   # For head axis broadcasting.

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
        # if hid_dim is not None,then query,key,value hid_dim is hid_dim
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
        self.fc_2 = nn.Linear(pf_dim, hid_dim)  # position-wise
        self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
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
            pf_dim, dropout=0.1, max_len=config.max_seq_length,
             scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Embedding(
            input_dim, hid_dim, padding_idx=config.PAD_idx)
        self.position_enc = LinearPosionEncoding(max_len=max_len,hid_dim=hid_dim)
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
            pf_dim, max_len=config.max_seq_length, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(
            output_dim, hid_dim, padding_idx=config.PAD_idx)
        self.position_enc = LinearPosionEncoding(max_len=max_len,hid_dim=hid_dim)
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

class Generator(LightningModule):
    def __init__(self, hid_dim, output_dim,dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(hid_dim, output_dim)
        # self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.proj(x)
            # x=self.dropout(x)
        return F.log_softmax(x, dim=-1)

class EMF(LightningModule):
    def __init__(self, vocab: Lang):
        super(EMF, self).__init__()
        self.vocab = vocab
        # self.encoder = Encoder()
        # embdim in most cases is the same as hid_dim
        #'emb': multiply \sqrt{d_model} to embedding output
        #'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        
        self.encoder = Encoder(
            input_dim=vocab.n_words,
            hid_dim=config.hidden_dim,
            n_layers=config.enc_layers,
            n_head=config.heads,
            pf_dim=config.pf_dim,
            dropout=config.dropout,
            scale_emb=config.scale_emb
        )

        self.decoder = Decoder(
            output_dim=vocab.n_words,
            hid_dim=config.hidden_dim,
            n_layers=config.dec_layers,
            n_head=config.heads,
            pf_dim=config.pf_dim,
            dropout=config.dropout,
            scale_emb=config.scale_emb
        )
        self.generator = Generator(config.hidden_dim, vocab.n_words,dropout=config.dropout)
    
        #if config.share_emb:
        #    self.trg_word_prj.weight = self.encoder.src_word_emb.weight
        #    self.x_logit_scale = (config.hidden_dim ** -0.5)
        #else:
        #self.x_logit_scale = 1

    def make_src_mask(self, src,batch_first=True):
        #src = [batch size, src len]
        src_mask = (src != config.PAD_idx).unsqueeze(1).unsqueeze(2)
        #src_mask = [batch size, 1, 1, src len]
        return src_mask
    
    def make_trg_mask(self, trg):
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != config.PAD_idx).unsqueeze(1).unsqueeze(2)
        #trg_pad_mask = [batch size, 1, 1, trg len]
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        #trg_sub_mask = [trg len, trg len]
        trg_mask = trg_pad_mask & trg_sub_mask
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, input_batch, target_batch, target_program):
        # divide data
        ref_batch = target_batch[:, 1:]
        target_batch = target_batch[:, :-1]
        src_mask = self.make_src_mask(input_batch)
        trg_mask = self.make_trg_mask(target_batch)
        # encode
        enc_output,*_= self.encoder(input_batch, src_mask)
        dec_output,*_= self.decoder(target_batch, trg_mask, enc_output, src_mask)

        output = self.generator(dec_output)
        loss = F.nll_loss(output.contiguous().view(-1, output.size(-1)),
                          ref_batch.contiguous().view(-1),
                          ignore_index=config.PAD_idx)
        return loss

    def init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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
        trg_tensor = self.decoder_greedy(input_batch, max_length=1000)
        trg_tensor = trg_tensor.squeeze().cpu().numpy()
        decoded_words = [self.vocab.index2word[idx.item()] for idx in trg_tensor]
        
        return decoded_words,batch['input_txt'][0],batch['target_txt'][0],batch['program_txt'][0],batch['situation_txt'][0]

    def test_epoch_end(self, outputs) -> None:
        file_path = f'./predicts/{config.model}-{config.emotion_emb_type}-results.txt'
        # print(outputs)
        with open(file_path,'w') as f:
            for (predict,contexts,ref,emotion,situation) in outputs:
                predict = ' '.join(predict)
                context = [' '.join(context) for context in contexts]
                ref = ' '.join(ref)
                situation = ' '.join(situation)
                f.write(f'context: {context})\n')
                f.write(f'emotion: {emotion})\n')
                f.write(f'situation: {situation})\n')
                f.write(f'ref: {ref})\n')
                f.write(f'predict: {predict})\n')
                f.write('----------------------------------------\n')




    def decoder_greedy(self, input_batch, max_length):
        #trg_tensor = [batch size, trg len - 1]
        trg_tensor = torch.zeros(
            (input_batch.size(0), max_length)).long().to(self.device)
        trg_tensor[:, 0] = config.SOS_idx

        src_mask = self.make_src_mask(input_batch)
        context,*_ = self.encoder(input_batch, src_mask)
        for i in range(1, max_length):
            trg_mask = self.make_trg_mask(trg_tensor)
            output,*_ = self.decoder(
               trg_tensor,trg_mask,context,src_mask)
            output = self.generator(output)
            _, next_word = torch.max(output[:, -1], dim=1)
            trg_tensor[:, i] = next_word
            # output = output.argmax(dim=-1)
            # trg_tensor[:, i] = output[:, i]
            if next_word == config.EOS_idx:
                break
            
            
        return trg_tensor

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
