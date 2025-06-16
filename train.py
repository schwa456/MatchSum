import argparse
import hydra
from src import *

parser = argparse.ArgumentParser()
parser.add_argument("--config-name", dest="config_name", default=None, type=str)
args = parser.parse_args()

@hydra.main(version_base=None, config_path='./config', config_name=args.config_name)
def train(cfg: DictConfig):
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    # Prepare Tokenizer, Model and Loss Function
    model = MatchSum(**cfg.model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_checkpoint)

    # Load Train and Validation Datasets
    train_df, val_df = get_train_df(cfg.dataset.path, cfg.dataset.use_df, **cfg.dataset.df)

    train_dataset = MatchSum_Dataset(train_df, tokenizer, cfg.max_seq_len)
    val_dataset = MatchSum_Dataset(val_df, tokenizer, cfg.max_seq_len)
    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, cfg.batch_size, shuffle=False)

    # Config Training
    engine = MatchSum_Engine(model, train_df, val_df, **cfg.engine)
    logger = My_WandbLogger(**cfg.log, save_artifact=False)
    cfg_trainer = Config_Trainer(cfg.trainer)()

    # Run Training
    trainer = pl.Trainer(
        **cfg_trainer,
        logger=logger,
        num_sanity_val_steps=0
    )
    logger.watch(engine)

    if cfg.train_checkpoint:
        trainer.fit(engine, train_loader, val_loader, ckpt_path=cfg.train_checkpoint)
    else:
        trainer.fit(engine, train_loader, val_loader)
    
    wandb.finish()

if __name__ == '__main__':
    train()