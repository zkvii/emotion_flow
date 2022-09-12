import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from util import config
from pytorch_lightning import LightningModule
from model.common import (
    share_embedding,
    LabelSmoothing,
    get_input_from_batch,
    get_output_from_batch,
    top_k_top_p_filtering,
)
from model.moel import Encoder, Generator
from model.MIME.emotion_input_attention import EmotionInputEncoder
from model.MIME.complex_res_attention import ComplexResDecoder
from model.MIME.complex_res_gate import ComplexResGate
from model.MIME.decoder_context_v import DecoderContextV
from model.MIME.VAE_noEmo_posterior import VAESampling
from model.translator.mime_translator import Translator

class MIME(LightningModule):
    """
    for emotion attention, simply pass the randomly sampled emotion as the Q in a decoder block of transformer
    """

    def __init__(
        self,
        vocab,
        decoder_number,
    ):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        self.embedding = share_embedding(self.vocab, config.pretrain_emb)

        self.encoder = Encoder(
            config.emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
        )
        self.decoder_number = decoder_number

        self.decoder = DecoderContextV(
            config.emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
        )

        self.vae_sampler = VAESampling(
            config.hidden_dim, config.hidden_dim, out_dim=300
        )

        # outputs m
        self.emotion_input_encoder_1 = EmotionInputEncoder(
            config.emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
            emo_input=config.emo_input,
        )
        # outputs m~
        self.emotion_input_encoder_2 = EmotionInputEncoder(
            config.emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
            emo_input=config.emo_input,
        )

        if config.emo_combine == "att":
            self.cdecoder = ComplexResDecoder(
                config.emb_dim,
                config.hidden_dim,
                num_layers=config.hop,
                num_heads=config.heads,
                total_key_depth=config.depth,
                total_value_depth=config.depth,
                filter_size=config.filter,
                universal=config.universal,
            )

        elif config.emo_combine == "gate":
            self.cdecoder = ComplexResGate(config.emb_dim)

        self.s_weight = nn.Linear(config.hidden_dim, config.emb_dim, bias=False)
        self.decoder_key = nn.Linear(config.hidden_dim, decoder_number, bias=False)

        # v^T tanh(W E[i] + H c + b)
        method3 = True
        if method3:
            self.e_weight = nn.Linear(config.emb_dim, config.emb_dim, bias=True)
            self.v = torch.rand(config.emb_dim, requires_grad=True)

        self.generator = Generator(config.hidden_dim, self.vocab_size)
        self.emoji_embedding = nn.Embedding(32, config.emb_dim)
        if config.init_emo_emb:
            self.init_emoji_embedding_with_glove()

        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if config.label_smoothing:
            self.criterion = LabelSmoothing(
                size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1
            )
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        if config.softmax:
            self.attention_activation = nn.Softmax(dim=1)
        else:
            self.attention_activation = nn.Sigmoid()  # nn.Softmax()

        self.positive_emotions = [11, 16, 6, 8, 3, 1, 28, 13, 31, 17, 24, 0, 27]
        self.negative_emotions = [
            9,
            4,
            2,
            22,
            14,
            30,
            29,
            25,
            15,
            10,
            23,
            19,
            18,
            21,
            7,
            20,
            5,
            26,
            12,
        ]


        self.res={}
        self.gdn={}
    def init_emoji_embedding_with_glove(self):
        self.emotions = [
            "surprised",
            "excited",
            "annoyed",
            "proud",
            "angry",
            "sad",
            "grateful",
            "lonely",
            "impressed",
            "afraid",
            "disgusted",
            "confident",
            "terrified",
            "hopeful",
            "anxious",
            "disappointed",
            "joyful",
            "prepared",
            "guilty",
            "furious",
            "nostalgic",
            "jealous",
            "anticipating",
            "embarrassed",
            "content",
            "devastated",
            "sentimental",
            "caring",
            "trusting",
            "ashamed",
            "apprehensive",
            "faithful",
        ]
        self.emotion_index = [self.vocab.word2index[i] for i in self.emotions]
        self.emoji_embedding_init = self.embedding(
            torch.Tensor(self.emotion_index).long()
        )
        self.emoji_embedding.weight.data = self.emoji_embedding_init
        self.emoji_embedding.weight.requires_grad = True

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
        # loss, ppl, bce, acc, top_preds, comet_res= self.train_one_batch(batch,batch_idx)
        loss, ppl, bce, acc = self.train_one_batch(batch, batch_idx)
        file_path=f'./predicts/{config.model}-{config.emotion_emb_type}-results.txt'
        outputs = open(file_path, 'a+', encoding='utf-8')
        self.log('test_ppl',ppl)
        self.log('test_loss',loss)
        self.log('test_bce',bce)
        self.log('test_acc',acc)
        sent_g=self.decoder_greedy(batch)
        t=Translator(self,self.vocab)
        sent_b = t.beam_search(batch,config.max_dec_step)
        ref, hyp_g= [], []
        for i, greedy_sent in enumerate(sent_g):
                rf = " ".join(batch["target_txt"][i])
                hyp_g.append(greedy_sent)
                ref.append(rf)
                self.res[batch_idx] = greedy_sent.split()
                self.gdn[batch_idx] = batch["target_txt"][i]  # targets.split()
                outputs.write(f"Emotion:{batch['program_txt'][i]} \n")
                outputs.write(f"Context:{[' '.join(s) for s in batch['input_txt'][i]]} \n")
                # outputs.write("Concept:{} \n".format(batch["concept_txt"]))
                outputs.write(f"Pred:{greedy_sent} \n")
                outputs.write(f"Beam:{sent_b[i]} \n")
                outputs.write(f"Ref:{rf} \n")

        return loss

    def random_sampling(self, e):
        p = np.random.choice(self.positive_emotions)
        n = np.random.choice(self.negative_emotions)
        if e in self.positive_emotions:
            mimic = p
            mimic_t = n
        else:
            mimic = n
            mimic_t = p
        return mimic, mimic_t

    def train_one_batch(self, batch, iter, train=True):
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


        ## Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)

        emb_mask = self.embedding(batch["input_mask"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src)

        q_h = (
            torch.mean(encoder_outputs, dim=1)
            if config.mean_query
            else encoder_outputs[:, 0]
        )
        # q_h = torch.max(encoder_outputs, dim=1)
        (
            emotions_mimic,
            emotions_non_mimic,
            mu_positive_prior,
            logvar_positive_prior,
            mu_negative_prior,
            logvar_negative_prior,
        ) = self.vae_sampler(q_h, batch["program_label"], self.emoji_embedding)
        # KLLoss = -0.5 * (torch.sum(1 + logvar_n - mu_n.pow(2) - logvar_n.exp()) + torch.sum(1 + logvar_p - mu_p.pow(2) - logvar_p.exp()))

        m_out = self.emotion_input_encoder_1(
            emotions_mimic.unsqueeze(1), encoder_outputs, mask_src
        )
        m_tilde_out = self.emotion_input_encoder_2(
            emotions_non_mimic.unsqueeze(1), encoder_outputs, mask_src
        )
        if train:
            (
                emotions_mimic,
                emotions_non_mimic,
                mu_positive_posterior,
                logvar_positive_posterior,
                mu_negative_posterior,
                logvar_negative_posterior,
            ) = self.vae_sampler.forward_train(
                q_h,
                batch["program_label"],
                self.emoji_embedding,
                M_out=m_out.mean(dim=1),
                M_tilde_out=m_tilde_out.mean(dim=1),
            )
            KLLoss_positive = self.vae_sampler.kl_div(
                mu_positive_posterior,
                logvar_positive_posterior,
                mu_positive_prior,
                logvar_positive_prior,
            )
            KLLoss_negative = self.vae_sampler.kl_div(
                mu_negative_posterior,
                logvar_negative_posterior,
                mu_negative_prior,
                logvar_negative_prior,
            )
            KLLoss = KLLoss_positive + KLLoss_negative
        else:
            KLLoss_positive = self.vae_sampler.kl_div(
                mu_positive_prior, logvar_positive_prior
            )
            KLLoss_negative = self.vae_sampler.kl_div(
                mu_negative_prior, logvar_negative_prior
            )
            KLLoss = KLLoss_positive + KLLoss_negative

        if config.emo_combine == "att":
            v = self.cdecoder(encoder_outputs, m_out, m_tilde_out, mask_src)
        elif config.emo_combine == "gate":
            v = self.cdecoder(m_out, m_tilde_out)

        x = self.s_weight(q_h)

        # method2: E (W@c)
        logit_prob = torch.matmul(
            x, self.emoji_embedding.weight.transpose(0, 1)
        )  # shape (b_size, 32)

        # Decode
        sos_token = (
            torch.LongTensor([config.SOS_idx] * enc_batch.size(0))
            .unsqueeze(1)
            
        ).to(self.device)
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1)

        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)

        pre_logit, attn_dist = self.decoder(
            self.embedding(dec_batch_shift), v, v, (mask_src, mask_trg)
        )

        ## compute output dist
        logit = self.generator(
            pre_logit,
            attn_dist,
            enc_batch_extend_vocab if config.pointer_gen else None,
            extra_zeros,
            attn_dist_db=None,
        )

        if train and config.schedule > 10:
            if random.uniform(0, 1) <= (
                0.0001 + (1 - 0.0001) * math.exp(-1.0 * iter / config.schedule)
            ):
                config.oracle = True
            else:
                config.oracle = False

        if config.softmax:
            program_label = torch.LongTensor(batch["program_label"]).to(self.device)

            if config.emo_combine == "gate":
                L1_loss = nn.CrossEntropyLoss()(logit_prob, program_label)
                loss = (
                    self.criterion(
                        logit.contiguous().view(-1, logit.size(-1)),
                        dec_batch.contiguous().view(-1),
                    )
                    + KLLoss
                    + L1_loss
                )
            else:
                L1_loss = nn.CrossEntropyLoss()(
                    logit_prob,
                    program_label,
                )
                loss = (
                    self.criterion(
                        logit.contiguous().view(-1, logit.size(-1)),
                        dec_batch.contiguous().view(-1),
                    )
                    + KLLoss
                    + L1_loss
                )

            loss_bce_program = nn.CrossEntropyLoss()(logit_prob, program_label)
        else:
            loss = self.criterion(
                logit.contiguous().view(-1, logit.size(-1)),
                dec_batch.contiguous().view(-1),
            ) + nn.BCEWithLogitsLoss()(
                logit_prob, torch.FloatTensor(batch["target_program"]).to(self.device)
            )
            loss_bce_program = nn.BCEWithLogitsLoss()(
                logit_prob, torch.FloatTensor(batch["target_program"]).to(self.device)
            )
        pred_program = np.argmax(logit_prob.detach().cpu().numpy(), axis=1)
        program_acc = accuracy_score(batch["program_label"], pred_program)

        if config.label_smoothing:
            loss_ppl = self.criterion_ppl(
                logit.contiguous().view(-1, logit.size(-1)),
                dec_batch.contiguous().view(-1),
            )


        if config.label_smoothing:
            return loss_ppl, math.exp(min(loss_ppl, 100)), loss_bce_program, program_acc
        else:
            return (
                loss,
                math.exp(min(loss.item(), 100)),
                loss_bce_program,
                program_acc,
            )

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30, emotion_classifier="built_in"):
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

        emotions = batch["program_label"]

        ## Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)

        emb_mask = self.embedding(batch["input_mask"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src)

        q_h = (
            torch.mean(encoder_outputs, dim=1)
            if config.mean_query
            else encoder_outputs[:, 0]
        )

        # method 2
        x = self.s_weight(q_h)
        logit_prob = torch.matmul(x, self.emoji_embedding.weight.transpose(0, 1))
        emo_pred = torch.argmax(logit_prob, dim=-1)

        if emotion_classifier == "vader":
            context_emo = [
                self.positive_emotions[0]
                if d["compound"] > 0
                else self.negative_emotions[0]
                for d in batch["context_emotion_scores"]
            ]
            context_emo = torch.Tensor(context_emo)
            (
                emotions_mimic,
                emotions_non_mimic,
                mu_p,
                logvar_p,
                mu_n,
                logvar_n,
            ) = self.vae_sampler(q_h, context_emo, self.emoji_embedding)
        elif emotion_classifier == None:
            (
                emotions_mimic,
                emotions_non_mimic,
                mu_p,
                logvar_p,
                mu_n,
                logvar_n,
            ) = self.vae_sampler(q_h, batch["program_label"], self.emoji_embedding)
        elif emotion_classifier == "built_in":
            (
                emotions_mimic,
                emotions_non_mimic,
                mu_p,
                logvar_p,
                mu_n,
                logvar_n,
            ) = self.vae_sampler(q_h, emo_pred, self.emoji_embedding)

        m_out = self.emotion_input_encoder_1(
            emotions_mimic.unsqueeze(1), encoder_outputs, mask_src
        )
        m_tilde_out = self.emotion_input_encoder_2(
            emotions_non_mimic.unsqueeze(1), encoder_outputs, mask_src
        )

        if config.emo_combine == "att":
            v = self.cdecoder(encoder_outputs, m_out, m_tilde_out, mask_src)
            # v = self.cdecoder(encoder_outputs, m_out, m_tilde_out, mask_src_chosen)
        elif config.emo_combine == "gate":
            v = self.cdecoder(m_out, m_tilde_out)
        elif config.emo_combine == "vader":
            context_emo_scores=batch['context_emotion_scores']
            m_weight = context_emo_scores.unsqueeze(-1).unsqueeze(-1)
            m_tilde_weight = 1 - m_weight
            v = m_weight * m_weight + m_tilde_weight * m_tilde_out

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(self.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(self.embedding(ys)),
                    self.embedding_proj_in(encoder_outputs),
                    self.embedding_proj_in(v),
                    (mask_src, mask_trg),
                    # attention_parameters,
                    config.depth
                )
            else:
                out, attn_dist = self.decoder(
                    self.embedding(ys), v, v, (mask_src, mask_trg)
                )

            logit = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            _, next_word = torch.max(logit[:, -1], dim=1)
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
                [ys, torch.ones(1, 1).long().fill_(next_word).to(self.device)], dim=1
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

    def decoder_topk(self, batch, max_dec_step=30, emotion_classifier="built_in"):
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

        emotions = batch["program_label"]

        context_emo = [
            self.positive_emotions[0]
            if d["compound"] > 0
            else self.negative_emotions[0]
            for d in batch["context_emotion_scores"]
        ]
        context_emo = torch.Tensor(context_emo)

        ## Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)

        emb_mask = self.embedding(batch["input_mask"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src)

        q_h = (
            torch.mean(encoder_outputs, dim=1)
            if config.mean_query
            else encoder_outputs[:, 0]
        )

        x = self.s_weight(q_h)
        # method 2
        logit_prob = torch.matmul(x, self.emoji_embedding.weight.transpose(0, 1))

        if emotion_classifier == "vader":
            context_emo = [
                self.positive_emotions[0]
                if d["compound"] > 0
                else self.negative_emotions[0]
                for d in batch["context_emotion_scores"]
            ]
            context_emo = torch.Tensor(context_emo)
            (
                emotions_mimic,
                emotions_non_mimic,
                mu_p,
                logvar_p,
                mu_n,
                logvar_n,
            ) = self.vae_sampler(q_h, context_emo, self.emoji_embedding)
        elif emotion_classifier == None:
            (
                emotions_mimic,
                emotions_non_mimic,
                mu_p,
                logvar_p,
                mu_n,
                logvar_n,
            ) = self.vae_sampler(q_h, batch["program_label"], self.emoji_embedding)
        elif emotion_classifier == "built_in":
            emo_pred = torch.argmax(logit_prob, dim=-1)
            (
                emotions_mimic,
                emotions_non_mimic,
                mu_p,
                logvar_p,
                mu_n,
                logvar_n,
            ) = self.vae_sampler(q_h, emo_pred, self.emoji_embedding)

        m_out = self.emotion_input_encoder_1(
            emotions_mimic.unsqueeze(1), encoder_outputs, mask_src
        )
        m_tilde_out = self.emotion_input_encoder_2(
            emotions_non_mimic.unsqueeze(1), encoder_outputs, mask_src
        )

        if config.emo_combine == "att":
            v = self.cdecoder(encoder_outputs, m_out, m_tilde_out, mask_src)
        elif config.emo_combine == "gate":
            v = self.cdecoder(m_out, m_tilde_out)
        elif config.emo_combine == "vader":

            context_emo_scores=batch['context_emotion_scores']
            m_weight = context_emo_scores.unsqueeze(-1).unsqueeze(-1)
            m_tilde_weight = 1 - m_weight
            v = m_weight * m_weight + m_tilde_weight * m_tilde_out

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(self.embedding(ys)),
                    self.embedding_proj_in(encoder_outputs),
                    (mask_src, mask_trg),
                    # attention_parameters,
                    config.depth
                )
            else:
                out, attn_dist = self.decoder(
                    self.embedding(ys), v, v, (mask_src, mask_trg)
                )

            logit = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
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
            next_word = next_word.data.item()

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word)], dim=1
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