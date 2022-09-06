# from turtle import forward
import torch
from pytorch_lightning import Trainer
# from tqdm import tqdm
from dataloader.loader import prepare_data_seq
# from torch.utils.data import DataLoader
from model.empdg import EMPDG
from model.moel import MOEL
from model.trans import Transformer
from model.MIME.model import MIME
# from model.moel import MOEL
# from model.trans import Transformer

from util import config
from model.litcem import CEM
from torch.nn.init import xavier_normal_
import torch.nn as nn
from torch.utils.data import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint,Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.core.saving import save_hparams_to_yaml
import os
os.environ['CUDA_VISIBLE_DEVICES'] = config.devices
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

logger = TensorBoardLogger(
    "em_logs", name=f"{config.model}", version=f'{config.emotion_emb_type}')


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
    elif config.model == 'empdg':
        model = EMPDG(
            vocab,
            decoder_number=decoder_num,
        )
    elif config.model == 'moel':
        model = MOEL(
            vocab,
            decoder_number=decoder_num,
        )
    elif config.model == 'trans':
        model = Transformer(
            vocab,
            decoder_number=decoder_num,
        )
    elif config.model == 'mult':
        model = Transformer(
            vocab,
            is_multitask=True,
            decoder_number=decoder_num,
        )
    elif config.model == 'mime':
        model = MIME(
            vocab,
            decoder_number=decoder_num,
        )

    # Intialization
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
            xavier_normal_(p)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_ppl", filename=f"{config.model}-{config.emotion_emb_type}", mode="min")

    class onCheckPointHparams(Callback):
        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            if trainer.current_epoch == 0:
                file_path = f"{trainer.logger.log_dir}/hparams.yaml"
                print(f"Saving hparams to file_path: {file_path}")
                save_hparams_to_yaml(config_yaml=file_path, hparams=pl_module.hparams)

    trainer = Trainer(
        max_epochs=config.max_epoch,
        accelerator='gpu',
        callbacks=[checkpoint_callback,onCheckPointHparams()],
        # progress_bar_refresh_rate=10
        logger=logger
    )
    # trainer_test=Trainer(accelerator='gpu',checkpoint_callback=False,logger=False)

    checkpoint_path = f'./em_logs/{config.model}/{config.emotion_emb_type}/checkpoints/{config.model}-{config.emotion_emb_type}.ckpt'
    # trainer.fit(model=model, train_dataloaders=train_loader)
    if config.mode == 'only_train':
        trainer.fit(model=model, train_dataloaders=train_loader,
                    val_dataloaders=dev_loader)
    elif config.mode == 'train_and_test':
        trainer.fit(model=model, train_dataloaders=train_loader,
                    val_dataloaders=dev_loader)
        print('--------------------start test---------------------')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        trainer.test(model=model, dataloaders=test_loader)
    else:
        print('--------------------start test---------------------')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        # print(model)
        # model.load_from_checkpoint(checkpoint_path)
        # clear old predicts if need overwrite use date
        file_path = f'./predicts/{config.model}-{config.emotion_emb_type}-results.txt'
        os.remove(file_path)
        trainer.test(model=model, dataloaders=test_loader)


if __name__ == '__main__':
    # prepare_data_seq()
    main()
