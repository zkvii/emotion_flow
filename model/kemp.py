import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from model.common import EncoderLayer, DecoderLayer, LabelSmoothing, _gen_bias_mask, _get_attn_subsequent_mask, share_embedding, LayerNorm, _gen_timing_signal
from pytorch_lightning import LightningModule
# from numpy import random
from sklearn.metrics import accuracy_score
from util import config
from model.translator.kemp_translator import Translator


class Encoder():
    """
    A Transformer Encoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, universal=False, concept=False):
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

        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if(self.universal):
            # for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length) if use_mask else None,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        self.embedding_proj = nn.Linear(
            embedding_size, hidden_size, bias=False)
        if(self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params)
                                     for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if(self.universal):
            if(config.act):  # Adaptive Computation Time
                x, (self.remainders, self.n_updates) = self.act_fn(
                    x, inputs, self.enc, self.timing_signal, self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:,
                                            :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(
                        1, inputs.shape[1], 1).type_as(inputs.data)
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:,
                                    :inputs.shape[1], :].type_as(inputs.data)

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

    def __init__(self,embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, universal=False):
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

        if(self.universal):
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
                  relu_dropout)

        if(self.universal):
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(*[DecoderLayer(*params)
                                     for l in range(num_layers)])

        self.embedding_proj = nn.Linear(
            embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_loss = nn.MSELoss()

    def forward(self, inputs, encoder_output, mask=None, pred_emotion=None, emotion_contexts=None, context_vad=None):
        '''
        inputs: (bsz, tgt_len)
        encoder_output: (bsz, src_len), src_len=dialog_len+concept_len
        mask: (bsz, src_len)
        pred_emotion: (bdz, emotion_type)
        emotion_contexts: (bsz, emb_dim)
        context_vad: (bsz, src_len) emotion intensity values
        '''
        mask_src, mask_trg = mask
        dec_mask = torch.gt(
            mask_trg.bool() + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)].bool(), 0)
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)
        loss_att = 0.0
        attn_dist = None
        if(self.universal):
            if(config.act):
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(
                    x, inputs, self.dec, self.timing_signal, self.position_signal, self.num_layers, encoder_output, decoding=True)
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:,
                                        :inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(
                        1, inputs.shape[1], 1).type_as(inputs.data)
                    x, _, pred_emotion, emotion_contexts, attn_dist, _ = self.dec(
                        (x, encoder_output, pred_emotion, emotion_contexts, [], (mask_src, dec_mask)))
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:,
                                    :inputs.shape[1], :].type_as(inputs.data)

            # Run decoder  y, encoder_outputs, pred_emotion, emotion_contexts, attention_weight, mask
            y, _, pred_emotion, emotion_contexts, attn_dist, _ = self.dec(
                (x, encoder_output, pred_emotion, emotion_contexts, [], (mask_src, dec_mask)))

            # Emotional attention loss
            if context_vad is not None:
                src_attn_dist = torch.mean(attn_dist, dim=1)  # (bsz, src_len)
                loss_att = self.attn_loss(src_attn_dist, context_vad)

            # Final layer normalization
            y = self.layer_norm(y)

        return y, attn_dist, loss_att


class Generator(LightningModule):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.emo_proj = nn.Linear(2 * d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(self, x, pred_emotion=None, emotion_context=None, attn_dist=None, enc_batch_extend_vocab=None, extra_zeros=None, temp=1):
        # pred_emotion (bsz, 1, embed_dim);  emotion_context: (bsz, emb_dim)
        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        if emotion_context is not None:
            # emotion_context = emotion_context.unsqueeze(1).repeat(1, x.size(1), 1)
            pred_emotion = pred_emotion.repeat(1, x.size(1), 1)
            # (bsz, tgt_len, 2 emb_dim)
            x = torch.cat((x, pred_emotion), dim=2)
            logit = self.emo_proj(x)
        else:
            logit = self.proj(x)  # x: (bsz, tgt_len, emb_dim)

        if config.pointer_gen:
            vocab_dist = F.softmax(logit/temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist/temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat(
                [enc_batch_extend_vocab.unsqueeze(1)]*x.size(1), 1)  # extend for all seq

            if extra_zeros is not None:
                extra_zeros = torch.cat(
                    [extra_zeros.unsqueeze(1)] * x.size(1), 1)
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 2)
            # if beam_search:
            #     enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab_[0].unsqueeze(0)]*x.size(0),0) ## extend for all seq

            logit = torch.log(vocab_dist_.scatter_add(
                2, enc_batch_extend_vocab_, attn_dist_) + 1e-18)
            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class KEMP(LightningModule):
    def __init__(self, vocab, decoder_number):
        super(KEMP, self).__init__()
        self.vocab = vocab
        # word2index, word2count, index2word, n_words = vocab
        self.word2index = vocab.word2index
        self.word2count = vocab.word2count
        self.index2word = vocab.index2word
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(
            vocab, config.pretrain_emb)  # args, n_words, word2index
        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop,
                               num_heads=config.heads, total_key_depth=config.depth, total_value_depth=config.depth,
                               max_length=config.max_seq_length, filter_size=config.filter, universal=config.universal)

        self.map_emo = {0: 'surprised', 1: 'excited', 2: 'annoyed', 3: 'proud',
                        4: 'angry', 5: 'sad', 6: 'grateful', 7: 'lonely', 8: 'impressed',
                        9: 'afraid', 10: 'disgusted', 11: 'confident', 12: 'terrified',
                        13: 'hopeful', 14: 'anxious', 15: 'disappointed', 16: 'joyful',
                        17: 'prepared', 18: 'guilty', 19: 'furious', 20: 'nostalgic',
                        21: 'jealous', 22: 'anticipating', 23: 'embarrassed', 24: 'content',
                        25: 'devastated', 26: 'sentimental', 27: 'caring', 28: 'trusting',
                        29: 'ashamed', 30: 'apprehensive', 31: 'faithful'}

        # GRAPH
        self.dropout = config.dropout
        self.W_q = nn.Linear(config.emb_dim, config.emb_dim)
        self.W_k = nn.Linear(config.emb_dim, config.emb_dim)
        self.W_v = nn.Linear(config.emb_dim, config.emb_dim)
        self.graph_out = nn.Linear(config.emb_dim, config.emb_dim)
        self.graph_layer_norm = LayerNorm(config.hidden_dim)

        # emotional signal distilling
        self.identify = nn.Linear(config.emb_dim, decoder_number, bias=False)
        self.activation = nn.Softmax(dim=1)

        # multiple decoders
        self.emotion_embedding = nn.Linear(decoder_number, config.emb_dim)
        self.decoder = Decoder(config.emb_dim, hidden_size=config.hidden_dim,  num_layers=config.hop, num_heads=config.heads,
                               total_key_depth=config.depth, total_value_depth=config.depth, filter_size=config.filter, max_length=config.max_seq_length,)

        self.decoder_key = nn.Linear(
            config.hidden_dim, decoder_number, bias=False)
        self.generator = Generator(config.hidden_dim, self.vocab_size)
        if config.project:
            self.embedding_proj_in = nn.Linear(
                config.emb_dim, config.hidden_dim, bias=False)
        if config.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if config.label_smoothing:
            self.criterion = LabelSmoothing(
                size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        self.res = {}
        self.gdn = {}

    def concept_graph(self, context, concept, adjacency_mask):
        '''

        :param context: (bsz, max_context_len, embed_dim)
        :param concept: (bsz, max_concept_len, embed_dim)
        :param adjacency_mask: (bsz, max_context_len, max_context_len + max_concpet_len)
        :return:
        '''
        # target = self.W_sem_emo(context)  # (bsz, max_context_len, emb_dim)
        # concept = self.W_sem_emo(concept)
        target = context
        # (bsz, max_context_len + max_concept_len, emb_dim)
        src = torch.cat((target, concept), dim=1)

        # QK attention
        q = self.W_q(target)  # (bsz, tgt_len, emb_dim)
        # (bsz, src_len, emb_dim); (bsz, src_len, emb_dim)
        k, v = self.W_k(src), self.W_v(src)
        # batch matrix multiply (bsz, tgt_len, src_len)
        attn_weights_ori = torch.bmm(q, k.transpose(1, 2))

        adjacency_mask = adjacency_mask.bool()
        attn_weights_ori.masked_fill_(
            adjacency_mask,
            1e-24
        )  # mask PAD
        attn_weights = torch.softmax(
            attn_weights_ori, dim=-1)  # (bsz, tgt_len, src_len)

        # if torch.isnan(attn_weights).sum() != 0:
        #     pdb.set_trace()

        attn_weights = F.dropout(
            attn_weights, p=self.dropout, training=self.training)

        # weigted sum
        attn = torch.bmm(attn_weights, v)  # (bsz, tgt_len, emb_dim)
        attn = self.graph_out(attn)

        attn = F.dropout(attn, p=self.dropout, training=self.training)
        new_context = self.graph_layer_norm(target + attn)

        new_context = torch.cat((new_context, concept), dim=1)
        return new_context

    def train_one_batch(self, batch,batch_idx):
        enc_batch = batch["context_batch"]
        enc_batch_extend_vocab = batch["context_ext_batch"]
        enc_vad_batch = batch['context_vad']
        concept_input = batch["concept_batch"]  # (bsz, max_concept_len)
        concept_ext_input = batch["concept_ext_batch"]
        concept_vad_batch = batch['concept_vad_batch']

        oovs = batch["oovs"]
        max_oov_length = len(
            sorted(oovs, key=lambda i: len(i), reverse=True)[0])
        extra_zeros = Variable(torch.zeros(
            (enc_batch.size(0), max_oov_length))).to(config.device)

        dec_batch = batch["target_batch"]
        dec_ext_batch = batch["target_ext_batch"]

        ## Embedding - context
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(
            1)  # (bsz, src_len)->(bsz, 1, src_len)
        # dialogue state embedding
        emb_mask = self.embedding(batch["mask_context"])
        src_emb = self.embedding(enc_batch)+emb_mask
        src_vad = enc_vad_batch  # (bsz, len, 1)  emotion intensity values

        if config.model != 'wo_ECE':  # emotional context graph encoding
            if concept_input.size()[0] != 0:
                mask_con = concept_input.data.eq(
                    config.PAD_idx).unsqueeze(1)  # real mask
                con_mask = self.embedding(
                    batch["mask_concept"])  # kg embedding
                con_emb = self.embedding(concept_input)+con_mask

                # Knowledge Update
                # (bsz, context+concept, emb_dim)
                src_emb = self.concept_graph(
                    src_emb, con_emb, batch["adjacency_mask_batch"])
                # (bsz, 1, context+concept)
                mask_src = torch.cat((mask_src, mask_con), dim=2)
                src_vad = torch.cat(
                    (enc_vad_batch, concept_vad_batch), dim=1)  # (bsz, len)

        ## Encode - context & concept
        encoder_outputs = self.encoder(
            src_emb, mask_src)  # (bsz, src_len, emb_dim)

        # emotional signal distilling
        src_vad = torch.softmax(src_vad, dim=-1)
        emotion_context_vad = src_vad.unsqueeze(2)
        emotion_context_vad = emotion_context_vad.repeat(
            1, 1, config.emb_dim)  # (bsz, len, emb_dim)
        emotion_context = torch.sum(
            emotion_context_vad * encoder_outputs, dim=1)  # c_e (bsz, emb_dim)
        emotion_contexts = emotion_context_vad * encoder_outputs

        emotion_logit = self.identify(
            emotion_context)  # e_p (bsz, emotion_num)
        loss_emotion = nn.CrossEntropyLoss(reduction='sum')(
            emotion_logit, batch['emotion_label'])

        pred_emotion = np.argmax(emotion_logit.detach().cpu().numpy(), axis=1)
        emotion_acc = accuracy_score(
            batch["emotion_label"].cpu().numpy(), pred_emotion)

        # Decode
        sos_emb = self.emotion_embedding(
            emotion_logit).unsqueeze(1)  # (bsz, 1, emb_dim)
        dec_emb = self.embedding(dec_batch[:, :-1])  # (bsz, tgt_len, emb_dim)
        # (bsz, tgt_len, emb_dim)
        dec_emb = torch.cat((sos_emb, dec_emb), dim=1)

        mask_trg = dec_batch.data.eq(config.PAD_idx).unsqueeze(1)
        if "wo_EDD" in config.model:
            pre_logit, attn_dist, loss_attn = self.decoder(inputs=dec_emb,
                                                           encoder_output=encoder_outputs,
                                                           mask=(
                                                               mask_src, mask_trg),
                                                           pred_emotion=None,
                                                           emotion_contexts=None)
        else:
            pre_logit, attn_dist, loss_attn = self.decoder(inputs=dec_emb,
                                                           encoder_output=encoder_outputs,
                                                           mask=(
                                                               mask_src, mask_trg),
                                                           pred_emotion=None,
                                                           emotion_contexts=emotion_context,
                                                           context_vad=src_vad)

        # compute output dist
        if config.model != 'wo_ECE':  # emotional context graph encoding
            if concept_input.size()[0] != 0:
                enc_batch_extend_vocab = torch.cat(
                    (enc_batch_extend_vocab, concept_ext_input), dim=1)
        logit = self.generator(pre_logit, None, None, attn_dist,
                               enc_batch_extend_vocab if config.pointer_gen else None, extra_zeros)
        loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)),
                              dec_batch.contiguous().view(-1) if config.pointer_gen else dec_ext_batch.contiguous().view(-1))
        loss += loss_emotion
        if config.attn_loss and config.model != "wo_EDD":
            loss += (0.1 * loss_attn)

        loss_ppl = 0.0
        if config.label_smoothing:
            loss_ppl = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)),
                                          dec_batch.contiguous().view(-1) if config.pointer_gen else dec_ext_batch.contiguous().view(-1)).item()

        if torch.sum(torch.isnan(loss)) != 0:
            print('loss is NAN :(')

        if config.label_smoothing:
            return loss_ppl, math.exp(min(loss_ppl, 100)), loss_emotion.item(), emotion_acc
        else:
            return loss.item(), math.exp(min(loss.item(), 100)), 0, 0

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1)/p_t.size(1))/p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30):
        enc_batch_extend_vocab, extra_zeros = None, None
        enc_batch = batch["context_batch"]
        enc_vad_batch = batch['context_vad']
        enc_batch_extend_vocab = batch["context_ext_batch"]

        concept_input = batch["concept_batch"]  # (bsz, max_concept_len)
        concept_ext_input = batch["concept_ext_batch"]
        concept_vad_batch = batch['concept_vad_batch']
        oovs = batch["oovs"]
        max_oov_length = len(
            sorted(oovs, key=lambda i: len(i), reverse=True)[0])
        extra_zeros = Variable(torch.zeros(
            (enc_batch.size(0), max_oov_length))).to(config.device)

        ## Encode - context
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(
            1)  # (bsz, src_len)->(bsz, 1, src_len)
        emb_mask = self.embedding(batch["mask_context"])
        src_emb = self.embedding(enc_batch) + emb_mask
        src_vad = enc_vad_batch  # (bsz, len, 1)

        if config.model != 'wo_ECE':  # emotional context graph encoding
            if concept_input.size()[0] != 0:
                mask_con = concept_input.data.eq(
                    config.PAD_idx).unsqueeze(1)  # real mask
                con_mask = self.embedding(
                    batch["mask_concept"])  # dialogue state
                con_emb = self.embedding(concept_input) + con_mask

                # Knowledge Update
                src_emb = self.concept_graph(src_emb, con_emb,
                                             batch["adjacency_mask_batch"])  # (bsz, context+concept, emb_dim)
                # (bsz, 1, context+concept)
                mask_src = torch.cat((mask_src, mask_con), dim=2)

                src_vad = torch.cat(
                    (enc_vad_batch, concept_vad_batch), dim=1)  # (bsz, len)
        encoder_outputs = self.encoder(
            src_emb, mask_src)  # (bsz, src_len, emb_dim)

        # Identify
        src_vad = torch.softmax(src_vad, dim=-1)
        emotion_context_vad = src_vad.unsqueeze(2)
        emotion_context_vad = emotion_context_vad.repeat(
            1, 1, config.emb_dim)  # (bsz, len, emb_dim)
        emotion_context = torch.sum(
            emotion_context_vad * encoder_outputs, dim=1)  # c_e (bsz, emb_dim)
        emotion_contexts = emotion_context_vad * encoder_outputs

        emotion_logit = self.identify(emotion_context)  # (bsz, emotion_num)

        if concept_input.size()[0] != 0 and config.model != 'wo_ECE':
            enc_ext_batch = torch.cat(
                (enc_batch_extend_vocab, concept_ext_input), dim=1)
        else:
            enc_ext_batch = enc_batch_extend_vocab

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long()
        ys_emb = self.emotion_embedding(
            emotion_logit).unsqueeze(1)  # (bsz, 1, emb_dim)
        sos_emb = ys_emb
        if config.USE_CUDA:
            ys = ys.cuda()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step+1):
            if config.project:
                out, attn_dist, _ = self.decoder(self.embedding_proj_in(
                    ys_emb), self.embedding_proj_in(encoder_outputs), (mask_src, mask_trg))
            else:
                out, attn_dist, _ = self.decoder(inputs=ys_emb,
                                                 encoder_output=encoder_outputs,
                                                 mask=(mask_src, mask_trg),
                                                 pred_emotion=None,
                                                 emotion_contexts=emotion_context,
                                                 context_vad=src_vad)

            prob = self.generator(
                out, None, None, attn_dist, enc_ext_batch if config.pointer_gen else None, extra_zeros)
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.index2word[str(
                ni.item())] for ni in next_word.view(-1)])
            next_word = next_word.data[0]

            if config.use_cuda:
                ys = torch.cat(
                    [ys, torch.ones(1, 1).long().fill_(next_word).cuda()], dim=1)
                ys = ys.cuda()
                ys_emb = torch.cat((ys_emb, self.embedding(
                    torch.ones(1, 1).long().fill_(next_word).cuda())), dim=1)
            else:
                ys = torch.cat(
                    [ys, torch.ones(1, 1).long().fill_(next_word)], dim=1)
                ys_emb = torch.cat((ys_emb, self.embedding(
                    torch.ones(1, 1).long().fill_(next_word))), dim=1)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>':
                    break
                else:
                    st += e + ' '
            sent.append(st)
        return sent

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=config.lr)
