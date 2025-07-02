#import argparse
import hydra
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from src import *

#parser = argparse.ArgumentParser()
#parser.add_argument("--config-name", dest="config_name", default=None, type=str)
#args = parser.parse_args()

os.environ["HYDRA_FULL_ERROR"] = "1"

def custom_collate_fn(batch):
    def stack(name):
        return torch.stack([item[name] for item in batch])

    return {
        'doc_input_ids': stack('doc_input_ids'),
        'doc_attention_mask': stack('doc_attention_mask'),
        'doc_cls_indices': pad_sequence(
            [item['doc_cls_indices'] for item in batch],
            batch_first=True,
            padding_value=0
        ),
        'cand_input_ids': stack('cand_input_ids'),
        'cand_attention_mask': stack('cand_attention_mask'),
        'cand_cls_indices': pad_sequence(
            [item['cand_cls_indices'] for item in batch],
            batch_first=True,
            padding_value=0
        ),
        'label': stack('label')
    }

@hydra.main(version_base=None, config_path='./config', config_name='train_config')
def train(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer)
    encoder = load_scorer_model(cfg.model.tokenizer, **cfg.encoder)
    
    train_df = pd.read_csv('./data/train_df.csv')
    val_df = pd.read_csv('./data/val_df.csv')
    
    train_data = MatchSumDataset(train_df, tokenizer, cfg.max_seq_len)
    val_data = MatchSumDataset(val_df, tokenizer, cfg.max_seq_len)
    
    train_loader = DataLoader(train_data, cfg.batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_data, cfg.batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.trainer.Trainer.max_epochs
    
    
    matchsum_model = MatchSum(**cfg.model)
    
    engine_args = dict(cfg.engine)
    engine_args['num_training_steps'] = total_steps
    engine = MatchSum_Engine(matchsum_model, encoder, **engine_args)
    logger = My_WandbLogger(**cfg.log, save_artifact=False)
    cfg_trainer = Config_Trainer(cfg.trainer)()
    
    trainer = pl.Trainer(
        **cfg_trainer,
        logger=logger,
        num_sanity_val_steps=0
    )
    
    trainer.fit(engine, train_loader, val_loader)
    
    wandb.finish()

if __name__ == '__main__':
    train()