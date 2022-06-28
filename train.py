from turtle import forward
from typing import Any
from pytorch_lightning import Trainer
from tqdm import tqdm
from dataloader.loader import prepare_data_seq
from torch.utils.data import DataLoader
from model.empdg import EMPDG
from model.moel import MOEL
from model.trans import Transformer
# from model.moel import MOEL
# from model.trans import Transformer

from util import config
from model.litcem import CEM
from torch.nn.init import xavier_normal_
import torch.nn as nn
from torch.utils.data import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
    
import os
os.environ['CUDA_VISIBLE_DEVICES']=config.devices

def preprocess():
    train_loader, dev_loader, test_loader, vocab, decoder_num = prepare_data_seq(
        batch_size=config.batch_size
    )
    print('preprocess ok')
    

def main():
    """main func
    """
    # decoder_num is eq2 emo_num
    train_loader, dev_loader, test_loader, vocab, decoder_num = prepare_data_seq(
        batch_size=config.batch_size
    )

    if config.model == 'cem':
        model = CEM(
            vocab,
            decoder_number=decoder_num,
        )
    elif config.model=='empdg':
        model = EMPDG(
            vocab,
            decoder_number=decoder_num,
        )
    elif config.model=='moel':
        model = MOEL(
            vocab,
            decoder_number=decoder_num,
        )
    elif config.model=='trans':
        model = Transformer(
            vocab,
            decoder_number=decoder_num,
        )
        

    # Intialization
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
            xavier_normal_(p)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_ppl", filename=f"{config.model}-{config.emotion_emb_type}", mode="min")
    trainer=Trainer(
        max_epochs=12,
        accelerator='gpu',
        callbacks=[checkpoint_callback],
        # progress_bar_refresh_rate=10
        )
    trainer.fit(model=model,train_dataloaders=train_loader,val_dataloaders=dev_loader)
if __name__ == '__main__':
    # prepare_data_seq()
    main()
    