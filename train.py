# from turtle import forward
from util.common import cal_metric
import torch
from pytorch_lightning import Trainer
from util.eval import cal_one_model
# from tqdm import tqdm
from dataloader.loader import prepare_data_seq
# from torch.utils.data import DataLoader
from model.empdg import EMPDG
from model.kemp import KEMP
from model.moel import MOEL
from model.trans import Transformer
from model.MIME.model import MIME
# from model.moel import MOEL
# from model.trans import Transformer

from util import config
from model.litcem import CEM
from torch.nn.init import xavier_normal_
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
import os
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = config.devices
seed_everything(42)
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from util.common import load_best_path, save_best_hparams,save_best_path
logger = TensorBoardLogger(
    "em_logs", 
    name=f"{config.model}",
    version=f'{config.emotion_emb_type}'
    )


def print_opts(opts):
    """Prints the values of all command-line arguments."""
    print("=" * 80)
    print("Opts".center(80))
    print("-" * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print("{:>30}: {:<30}".format(key, opts.__dict__[key]).center(80))
    print("=" * 80)
    
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
    elif config.model == 'kemp':
        model = KEMP(vocab=vocab,decoder_number=decoder_num)
    # Intialization
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
            xavier_normal_(p)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_ppl", filename=f"{config.model}-{config.emotion_emb_type}", mode="min")

    # class onCheckPointHparams(Callback):
    #     def on_save_checkpoint(self, trainer, pl_module, checkpoint):
    #         if trainer.current_epoch == 0:
    #             file_path = f"{trainer.logger.log_dir}/hparams.yaml"
    #             print(f"Saving hparams to file_path: {file_path}")
    #             save_hparams_to_yaml(config_yaml=file_path, hparams=pl_module.hparams)

    trainer = Trainer(
        max_epochs=config.max_epoch,
        accelerator='gpu',
        # callbacks=[checkpoint_callback,onCheckPointHparams()],
        callbacks=[checkpoint_callback],
        # progress_bar_refresh_rate=10
        logger=logger
    )
    # trainer_test=Trainer(accelerator='gpu',checkpoint_callback=False,logger=False)
    print_opts(config.args)
    # checkpoint_path = f'./em_logs/{config.model}/{config.emotion_emb_type}/checkpoints/{config.model}-{config.emotion_emb_type}.ckpt'
    # trainer.fit(model=model, train_dataloaders=train_loader)
    if config.mode == 'only_train':
        trainer.fit(model=model, train_dataloaders=train_loader,
                    val_dataloaders=dev_loader)
        checkpoint_path=checkpoint_callback.best_model_path
        save_best_path(checkpoint_path)
        save_best_hparams(model)
    elif config.mode == 'train_and_test':
        trainer.fit(model=model, train_dataloaders=train_loader,
                    val_dataloaders=dev_loader)
        print('--------------------start test---------------------')
        #save and load
        checkpoint_path=checkpoint_callback.best_model_path
        save_best_path(checkpoint_path)
        save_best_hparams(model)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        #clear old predicts
        file_path = f'./predicts/{config.model}-{config.emotion_emb_type}-results.txt'
        if os.path.exists(file_path):
            os.remove(file_path)
        trainer.test(model=model, dataloaders=test_loader)

    else:
        print('--------------------start test---------------------')
        #load best model
        checkpoint_path=load_best_path()
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        # clear old predicts if need overwrite use date
        file_path = f'./predicts/{config.model}-{config.emotion_emb_type}-results.txt'
        if os.path.exists(file_path):
            os.remove(file_path)
        trainer.test(model=model, dataloaders=test_loader)

if __name__ == '__main__':
    if config.preprocess:
        prepare_data_seq()
    else:
        main()
    # cal_metric('./predicts/trans-origin-results.txt')
