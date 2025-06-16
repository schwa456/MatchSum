import argparse
import hydra
from src import *

parser = argparse.ArgumentParser()
parser.add_argument("--config-name", dest='config_name', default=None, type=str)
args = parser.parse_args()


@hydra.main(version_base=None, config_path='./config/', config_name=args.config_name)
def test(cfg: DictConfig):
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    model = MatchSum(**cfg.model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_checkpoint)

    # load food datasets
    inference_df = pd.read_pickle(os.path.join(cfg.inference.path, "food_data_2023.pkl"))
        
    inference_dataset = MatchSum_Dataset(inference_df, tokenizer, cfg.max_seq_len)
    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

    # engine = ExtSum_Engine(model, test_df=test_df, sum_size=3, n_block=3, **cfg.engine)
    engine = MatchSum_Engine(model, inference_df=inference_df, **cfg.engine)
    cfg_trainer = Config_Trainer(cfg.trainer)()
    trainer = pl.Trainer(**cfg_trainer, logger=False)

    from torch.serialization import add_safe_globals
    from omegaconf import ListConfig, DictConfig, AnyNode 
    from omegaconf.base import ContainerMetadata, Metadata
    from typing import Any
    from collections import defaultdict
    add_safe_globals([ListConfig, ContainerMetadata, Any, list, defaultdict, dict, int, DictConfig, AnyNode, Metadata])
    
    if 'test_checkpoint' in cfg:
        trainer.predict(engine, inference_loader, ckpt_path=cfg.test_checkpoint)
    else:
        raise RuntimeError('no checkpoint is given')


if __name__ == "__main__":
    test()
