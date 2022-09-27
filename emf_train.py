from util.common import load_best_path, save_best_hparams, save_best_path
from model.emf import EMF
from util.common import cal_metric
import torch
from pytorch_lightning import Trainer
from util.eval import cal_one_model
# from tqdm import tqdm
from dataloader.emf_loader import prepare_data_seq

from util import config
from torch.nn.init import xavier_normal_
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = config.devices

logger = TensorBoardLogger(
    "emf_logs",
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
    train_loader, dev_loader, test_loader, vocab= prepare_data_seq(
        batch_size=config.batch_size
    )
    sample= next(iter(train_loader))
    
    print('preprocess ok')


def main():
    """main func
    """
    train_loader, dev_loader, test_loader, vocab, decoder_num = prepare_data_seq(
        batch_size=config.batch_size
    )

    model = EMF(vocab=vocab)
    # Intialization
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
            xavier_normal_(p)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_ppl", filename=f"{config.model}-{config.emotion_emb_type}", mode="min")


    trainer = Trainer(
        max_epochs=config.max_epoch,
        accelerator='gpu',
        callbacks=[checkpoint_callback],
        logger=logger
    )
    print_opts(config.args)

    if config.mode == 'only_train':
        trainer.fit(model=model, train_dataloaders=train_loader,
                    val_dataloaders=dev_loader)
        checkpoint_path = checkpoint_callback.best_model_path
        save_best_path(checkpoint_path)
        save_best_hparams(model)
    elif config.mode == 'train_and_test':
        trainer.fit(model=model, train_dataloaders=train_loader,
                    val_dataloaders=dev_loader)
        print('--------------------start test---------------------')
        #save and load
        checkpoint_path = checkpoint_callback.best_model_path
        save_best_path(checkpoint_path)
        save_best_hparams(model)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        # clear old predicts
        file_path = f'./predicts/{config.model}-{config.emotion_emb_type}-results.txt'
        if os.path.exists(file_path):
            os.remove(file_path)
        trainer.test(model=model, dataloaders=test_loader)

    else:
        print('--------------------start test---------------------')
        # load best model
        checkpoint_path = load_best_path()
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        # clear old predicts if need overwrite use date
        file_path = f'./predicts/{config.model}-{config.emotion_emb_type}-results.txt'
        if os.path.exists(file_path):
            os.remove(file_path)
        trainer.test(model=model, dataloaders=test_loader)

if __name__ == '__main__':
    if config.preprocess:
        preprocess()
    else:
        main()
