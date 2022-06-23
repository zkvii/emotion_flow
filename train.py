from turtle import forward
from typing import Any
from pytorch_lightning import Trainer
from tqdm import tqdm
from dataloader.loader import prepare_data_seq
from torch.utils.data import DataLoader

from util import config
from model.litcem import CEM
from torch.nn.init import xavier_normal_
import torch.nn as nn
from torch.utils.data import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'


def main():
    """main func
    """
    # decoder_num is eq2 emo_num
    train_set, dev_set, test_set, vocab, decoder_num = prepare_data_seq(
        batch_size=config.batch_size
    )

    model = CEM(
        vocab,
        decoder_number=decoder_num,
        is_eval=config.test,
        model_file_path=config.model_path if config.test else None
    ).to('cuda')

    # Intialization
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
            xavier_normal_(p)
    
    dataloader=DataLoader(dataset=train_set,batch_size=config.batch_size,shuffle=True)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_ppl", filename=f"{config.model}", mode="min")
    trainer=Trainer(
        max_epochs=12,
        accelerator='gpu',
        callbacks=[checkpoint_callback],
        progress_bar_refresh_rate=100
        )
    trainer.fit(model=model,train_dataloaders=dataloader)
if __name__ == '__main__':
    main()
    