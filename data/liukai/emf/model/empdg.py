import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from pytorch_lightning import LightningModule
from model.common import (
    EncoderLayer,
    DecoderLayer,
    LayerNorm,
    _gen_bias_mask,
    _gen_timing_signal,
    share_embedding,
    LabelSmoothing,
    get_input_from_batch,
    get_output_from_batch,
    _get_attn_subsequent_mask,
)
from util import config

from sklearn.metrics import accuracy_score


class Semantic_Encoder(LightningModule):
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
        concept=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder  2
            num_heads: Number of attention heads   2
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head   40
            total_value_depth: Size of last dimension of values. Must be divisible by num_head  40
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN  50
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Semantic_Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
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

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if self.universal:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

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
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Emotion_Encoder(LightningModule):
    """
    A Transformer Encoder module.
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
        concept=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder  2
            num_heads: Number of attention heads   2
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head   40
            total_value_depth: Size of last dimension of values. Must be divisible by num_head  40
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN  50
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Emotion_Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
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

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if self.universal:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

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
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(LightningModule):
    """
    A Transformer Decoder module.
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
            ## for t
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

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask=None):
        mask_src, mask_trg = mask
        mask_src=mask_src.to(self.device)
        mask_trg=mask_trg.to(self.device)
        dec_mask = torch.gt(
            mask_trg.bool()
            + self.mask[:, : mask_trg.size(-1), : mask_trg.size(-1)].bool().to(self.device),
            0,
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
                x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
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
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)

        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.emo_proj = nn.Linear(2 * d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        x,
        attn_dist=None,
        enc_batch_extend_vocab=None,
        max_oov_length=None,
        temp=1,
        beam_search=False,
        attn_dist_db=None,
    ):
        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)  # x: (bsz, tgt_len, emb_dim)

        if config.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat(
                [enc_batch_extend_vocab.unsqueeze(1)] * x.size(1), 1
            )  ## extend for all seq

            extra_zeros = Variable(torch.zeros((logit.size(0), max_oov_length)))
            if extra_zeros is not None:
                extra_zeros = torch.cat([extra_zeros.unsqueeze(1)] * x.size(1), 1)
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 2)

            logit = torch.log(
                vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_) + 1e-18
            )

            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class EMPDG(LightningModule):
    def __init__(
        self,
        vocab,
        decoder_number,
    ):
        """
        :param decoder_number: the number of emotion labels, i.e., 32
        """
        super(EMPDG, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        self.semantic_und = Semantic_Encoder(
            config.emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
        )
        self.emotion_pec = Emotion_Encoder(
            config.emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
        )
        self.map_emo = {
            0: "surprised",
            1: "excited",
            2: "annoyed",
            3: "proud",
            4: "angry",
            5: "sad",
            6: "grateful",
            7: "lonely",
            8: "impressed",
            9: "afraid",
            10: "disgusted",
            11: "confident",
            12: "terrified",
            13: "hopeful",
            14: "anxious",
            15: "disappointed",
            16: "joyful",
            17: "prepared",
            18: "guilty",
            19: "furious",
            20: "nostalgic",
            21: "jealous",
            22: "anticipating",
            23: "embarrassed",
            24: "content",
            25: "devastated",
            26: "sentimental",
            27: "caring",
            28: "trusting",
            29: "ashamed",
            30: "apprehensive",
            31: "faithful",
        }

        ## emotional signal distilling
        self.identify = nn.Linear(config.emb_dim, decoder_number, bias=False)
        self.identify_new = nn.Linear(2 * config.emb_dim, decoder_number, bias=False)
        self.activation = nn.Softmax(dim=1)

        ## decoders
        self.emotion_embedding = nn.Linear(decoder_number, config.emb_dim)
        self.decoder = Decoder(
            config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
        )

        self.decoder_key = nn.Linear(config.hidden_dim, decoder_number, bias=False)
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

        self.res={}
        self.gdn={}

    def training_step(self,batch,batch_idx):
        loss, ppl, bce, acc = self.train_one_batch(batch,batch_idx)
        self.log('train_ppl',ppl)
        self.log('train_loss',loss)
        self.log('train_bce',bce)
        self.log('train_acc',acc)
        return loss

    def validation_step(self,batch,batch_idx):
        loss, ppl, bce, acc = self.train_one_batch(batch,batch_idx)
        self.log('valid_ppl',ppl)
        self.log('valid_loss',loss)
        self.log('valid_bce',bce)
        self.log('valid_acc',acc)
        return loss
    def test_step(self,batch,batch_idx):
        loss, ppl, bce, acc= self.train_one_batch(batch,batch_idx)

        file_path=f'./predicts/{config.model}-{config.emotion_emb_type}-results.txt'
        outputs = open(file_path, 'a+', encoding='utf-8')
        self.log('test_ppl',ppl)
        self.log('test_loss',loss)
        self.log('test_bce',bce)
        self.log('test_acc',acc)
        sent_g=self.decoder_greedy(batch)
        ref, hyp_g= [], []
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
                outputs.write("Ref:{} \n".format(rf))

        return loss
    
    def train_one_batch(self, batch, iter, train=True, loss_from_d=0.0):
        enc_emo_batch = batch["program_context_batch"]

        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)

        mask_semantic = enc_batch.data.eq(config.PAD_idx).unsqueeze(
            1
        )  # (bsz, src_len)->(bsz, 1, src_len)
        sem_emb_mask = self.embedding(batch["input_mask"])  # dialogue state  E_d
        sem_emb = self.embedding(enc_batch) + sem_emb_mask  # E_w+E_d
        sem_encoder_outputs = self.semantic_und(
            sem_emb, mask_semantic
        )  # C_u  (bsz, sem_w_len, emb_dim)

        ## Multi-resolution Emotion Perception (understanding & predicting)
        mask_emotion = enc_emo_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emo_encoder_outputs = self.emotion_pec(
            self.embedding(enc_emo_batch), mask_emotion
        )  # C_e  (bsz, emo_w_len, emb_dim)

        emotion_logit = self.identify_new(
            torch.cat(
                (emo_encoder_outputs[:, 0, :], sem_encoder_outputs[:, 0, :]), dim=-1
            )
        )  # (bsz, decoder_number)
        emo_label = torch.LongTensor(batch["program_label"]).to(self.device)
        loss_emotion = nn.CrossEntropyLoss()(emotion_logit, emo_label)
        pred_emotion = np.argmax(emotion_logit.detach().cpu().numpy(), axis=1)
        emotion_acc = accuracy_score(batch["program_label"], pred_emotion)

        ## Combine Two Contexts
        src_emb = torch.cat(
            (sem_encoder_outputs, emo_encoder_outputs), dim=1
        )  # (bsz, src_len, emb_dim)
        mask_src = torch.cat((mask_semantic, mask_emotion), dim=2)  # (bsz, 1, src_len)

        ## Empathetic Response Generation
        sos_emb = self.emotion_embedding(emotion_logit).unsqueeze(1)
        dec_emb = self.embedding(dec_batch[:, :-1])
        dec_emb = torch.cat((sos_emb, dec_emb), dim=1)  # (bsz, 1+tgt_len, emb_dim)

        mask_trg = dec_batch.data.eq(config.PAD_idx).unsqueeze(1)
        # inputs, encoder_output, pred_emotion=None, emotion_contexts=None, mask=None
        pre_logit, attn_dist = self.decoder(dec_emb, src_emb, (mask_src, mask_trg))

        ## compute output dist
        logit = self.generator(
            pre_logit,
            attn_dist,
            enc_batch_extend_vocab if config.pointer_gen else None,
            extra_zeros,
            attn_dist_db=None,
        )
        ## loss: NNL if ptr else Cross entropy
        loss = self.criterion(
            logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1)
        )

        loss += loss_emotion
        loss += loss_from_d

        if config.label_smoothing:
            loss_ppl = self.criterion_ppl(
                logit.contiguous().view(-1, logit.size(-1)),
                dec_batch.contiguous().view(-1),
            )


        if config.label_smoothing:
            return (
                loss_ppl,
                math.exp(min(loss_ppl, 100)),
                loss_emotion.item(),
                emotion_acc,
            )
        else:
            return loss, math.exp(min(loss.item(), 100)), 0, 0

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        enc_emo_batch = batch["program_context_batch"]

        # if config.noam:
        #     self.optimizer.optimizer.zero_grad()
        # else:
        #     self.optimizer.zero_grad()

        ## Semantic Understanding
        mask_semantic = enc_batch.data.eq(config.PAD_idx).unsqueeze(
            1
        )  # (bsz, src_len)->(bsz, 1, src_len)
        sem_emb_mask = self.embedding(batch["input_mask"])  # dialogue state  E_d
        sem_emb = self.embedding(enc_batch) + sem_emb_mask  # E_w+E_d
        sem_encoder_outputs = self.semantic_und(
            sem_emb, mask_semantic
        )  # C_u  (bsz, sem_w_len, emb_dim)

        # Multi-resolution Emotion Perception (understanding & predicting)
        mask_emotion = enc_emo_batch.data.eq(config.PAD_idx).unsqueeze(1)
        # emo_emb_mask = self.embedding(batch["mask_emotion_context"])
        # emo_emb = self.embedding(enc_emo_batch) + emo_emb_mask
        emo_encoder_outputs = self.emotion_pec(
            self.embedding(enc_emo_batch), mask_emotion
        )  # C_e  (bsz, emo_w_len, emb_dim)

        ## Identify
        # emotion_logit = self.identify(emo_encoder_outputs[:,0,:])  # (bsz, decoder_number)
        emotion_logit = self.identify_new(
            torch.cat(
                (emo_encoder_outputs[:, 0, :], sem_encoder_outputs[:, 0, :]), dim=-1
            )
        )  # (bsz, decoder_number)

        ## Combine Two Contexts
        src_emb = torch.cat(
            (sem_encoder_outputs, emo_encoder_outputs), dim=1
        )  # (bsz, src_len, emb_dim)
        mask_src = torch.cat((mask_semantic, mask_emotion), dim=2)  # (bsz, 1, src_len)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long()
        ys_emb = self.emotion_embedding(emotion_logit).unsqueeze(1)  # (bsz, 1, emb_dim)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(ys_emb),
                    self.embedding_proj_in(src_emb),
                    (mask_src, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(ys_emb, src_emb, (mask_src, mask_trg))

            prob = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
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
                [ys.to(self.device), torch.ones(1, 1).long().fill_(next_word).to(self.device)], dim=1
            )
            ys_emb = torch.cat(
                (
                    ys_emb,
                    self.embedding(torch.ones(1, 1).long().fill_(next_word).to(self.device)),
                ),
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
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=config.lr)