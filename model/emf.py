# TAKEN FROM https://github.com/kolloldas/torchnlp
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from dataloader.loader import Lang
from model.common import (
    EncoderLayer,
    DecoderLayer,
    LayerNorm,
    _gen_bias_mask,
    _gen_timing_signal,
    get_input_from_batch,
    share_embedding,
    LabelSmoothing,
    _get_attn_subsequent_mask,
    get_output_from_batch,
    top_k_top_p_filtering,
)
from util import config
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score
from model.translator.emf_translator import Translator


class Encoder(LightningModule):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        use_mask=False,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            # for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length) if use_mask else None,
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        self.embedding_proj = nn.Linear(
            embedding_size, hidden_size, bias=False)
        if self.universal:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params)
                                     for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

        if config.act:
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.enc,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                )
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, : inputs.shape[1], :].type_as(
                        inputs.data
                    )
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:,
                                    : inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(LightningModule):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            # for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length),  # mandatory
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        if self.universal:
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(
                *[DecoderLayer(*params) for l in range(num_layers)]
            )

        self.embedding_proj = nn.Linear(
            embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(
            mask_trg +
            self.mask[:, : mask_trg.size(-1),
                      : mask_trg.size(-1)].to(mask_trg.device), 0
        )
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.dec,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                    encoder_output,
                    decoding=True,
                )
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:,
                                        : inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x, _, attn_dist, _ = self.dec(
                        (x, encoder_output, [], (mask_src, dec_mask))
                    )
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:,
                                    : inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec(
                (x, encoder_output, [], (mask_src, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist


class Generator(LightningModule):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(self, x):
        logit = self.proj(x)
        return F.log_softmax(logit, dim=-1)


class EMF(LightningModule):
    def __init__(
        self,
        vocab:Lang
    ):
        super(EMF, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        self.encoder = Encoder(
            config.emb_dim,
            config.hidden_dim,
            num_layers=config.enc_layers,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
        )

        # multiple decoders
        self.decoder = Decoder(
            config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.dec_layers,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
        )

        self.generator = Generator(config.hidden_dim, self.vocab_size)

        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if config.label_smoothing:
            self.criterion = LabelSmoothing(
                size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1
            )
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)
        self.res = {}
        self.gdn = {}

    def training_step(self, batch, batch_idx):
        loss, ppl, bce, acc = self.train_one_batch(batch, batch_idx)
        self.log('train_ppl', ppl)
        self.log('train_loss', loss)
        self.log('train_bce', bce)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ppl, bce, acc = self.train_one_batch(batch, batch_idx)
        self.log('valid_ppl', ppl)
        self.log('valid_loss', loss)
        self.log('valid_bce', bce)
        self.log('valid_acc', acc)
        return loss

    def test_step(self, batch, batch_idx):

        loss, ppl, bce, acc = self.train_one_batch(batch, batch_idx)

        file_path = f'./predicts/{config.model}-{config.emotion_emb_type}-results.txt'
        outputs = open(file_path, 'a+', encoding='utf-8')
        self.log('test_ppl', ppl)
        self.log('test_loss', loss)
        self.log('test_bce', bce)
        self.log('test_acc', acc)
        sent_g = self.decoder_greedy(batch)
        t = Translator(self, self.vocab)
        sent_b = t.beam_search(batch, max_dec_step=config.max_dec_step)
        ref, hyp_g = [], []
        for i, greedy_sent in enumerate(sent_g):
            rf = " ".join(batch["target_txt"][i])
            hyp_g.append(greedy_sent)
            ref.append(rf)
            self.res[batch_idx] = greedy_sent.split()
            self.gdn[batch_idx] = batch["target_txt"][i]  # targets.split()
            outputs.write("Emotion:{} \n".format(batch["program_txt"][i]))
            outputs.write("Context:{} \n".format(
                [" ".join(s) for s in batch['input_txt'][i]]))
            # outputs.write("Concept:{} \n".format(batch["concept_txt"]))
            outputs.write("Pred:{} \n".format(greedy_sent))
            outputs.write(f"Beam:{sent_b[i]} \n")
            outputs.write("Ref:{} \n".format(rf))

        return loss

    def train_one_batch(self, batch, iter, train=True):
        enc_batch = batch['input_batch']
        dec_batch = batch['target_batch']

        # if config.noam:
        #     self.optimizer.optimizer.zero_grad()
        # else:
        #     self.optimizer.zero_grad()

        # Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)

        emb_mask = self.embedding(batch["input_mask"])
        encoder_outputs = self.encoder(
            self.embedding(enc_batch) + emb_mask, mask_src)
        # Decode
        sos_token = (
            torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1)
        ).to(self.device)
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1)

        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        pre_logit, attn_dist = self.decoder(
            self.embedding(
                dec_batch_shift), encoder_outputs, (mask_src, mask_trg)
        )

        # compute output dist
        logit = self.generator(pre_logit)
        # logit = F.log_softmax(logit,dim=-1) #fix the name later
        # loss: NNL if ptr else Cross entropy
        loss = self.criterion(
            logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1)
        )
        return loss, math.exp(min(loss.item(), 100)), 0, 0

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30):
        enc_batch = batch['input_batch']
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["input_mask"])
        encoder_outputs = self.encoder(
            self.embedding(enc_batch) + emb_mask, mask_src)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(emb_mask.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(self.embedding(ys)),
                    self.embedding_proj_in(encoder_outputs),
                    (mask_src, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(
                    self.embedding(ys), encoder_outputs, (mask_src, mask_trg)
                )

            prob = self.generator(
                out
            )
            # logit = F.log_softmax(logit,dim=-1) #fix the name later
            # filtered_logit = top_k_top_p_filtering(logit[:, -1], top_k=0, top_p=0, filter_value=-float('Inf'))
            # Sample from the filtered distribution
            # next_word = torch.multinomial(F.softmax(filtered_logit, dim=-1), 1).squeeze()
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            next_word = next_word.data[0]

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(ys.device)],
                dim=1,
            )
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent

    def decoder_topk(self, batch, max_dec_step=30):
        enc_batch = batch["input_batch"]
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["input_mask"])
        encoder_outputs = self.encoder(
            self.embedding(enc_batch) + emb_mask, mask_src)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            out, attn_dist = self.decoder(
                self.embedding(ys), encoder_outputs, (mask_src, mask_trg)
            )

            logit = self.generator(
                out
            )
            filtered_logit = top_k_top_p_filtering(
                logit[:, -1], top_k=3, top_p=0, filter_value=-float("Inf")
            )
            # Sample from the filtered distribution
            next_word = torch.multinomial(
                F.softmax(filtered_logit, dim=-1), 1
            ).squeeze()
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            next_word = next_word.data[0]

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word)],
                dim=1,
            )

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=config.lr)


# CONVERTED FROM https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer_util.py#L1062
class ACT_basic(LightningModule):
    def __init__(self, hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size, 1)
        self.p.bias.data.fill_(1)
        self.threshold = 1 - 0.1

    def forward(
        self,
        state,
        inputs,
        fn,
        time_enc,
        pos_enc,
        max_hop,
        encoder_output=None,
        decoding=False,
    ):
        # init_hdd
        ## [B, S]
        halting_probability = torch.zeros(inputs.shape[0], inputs.shape[1]).to(
            self.device
        )
        # [B, S
        remainders = torch.zeros(inputs.shape[0], inputs.shape[1])
        ## [B, S]
        n_updates = torch.zeros(inputs.shape[0], inputs.shape[1])
        ## [B, S, HDD]
        previous_state = torch.zeros_like(inputs)

        step = 0
        # for l in range(self.num_layers):
        while (
            ((halting_probability < self.threshold) & (n_updates < max_hop))
            .byte()
            .any()
        ):
            # Add timing signal
            state = state + time_enc[:,
                                     : inputs.shape[1], :].type_as(inputs.data)
            state = state + pos_enc[:, step, :].unsqueeze(1).repeat(
                1, inputs.shape[1], 1
            ).type_as(inputs.data)

            p = self.sigma(self.p(state)).squeeze(-1)
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (
                halting_probability + p * still_running > self.threshold
            ).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (
                halting_probability + p * still_running <= self.threshold
            ).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            if decoding:
                state, _, attention_weight = fn((state, encoder_output, []))
            else:
                # apply transformation on the state
                state = fn(state)

            # update running part in the weighted state and keep the rest
            previous_state = (state * update_weights.unsqueeze(-1)) + (
                previous_state * (1 - update_weights.unsqueeze(-1))
            )
            if decoding:
                if step == 0:
                    previous_att_weight = torch.zeros_like(attention_weight).to(
                        config.device
                    )  # [B, S, src_size]
                previous_att_weight = (
                    attention_weight * update_weights.unsqueeze(-1)
                ) + (previous_att_weight * (1 - update_weights.unsqueeze(-1)))
            # previous_state is actually the new_state at end of hte loop
            # to save a line I assigned to previous_state so in the next
            # iteration is correct. Notice that indeed we return previous_state
            step += 1

        if decoding:
            return previous_state, previous_att_weight, (remainders, n_updates)
        else:
            return previous_state, (remainders, n_updates)
