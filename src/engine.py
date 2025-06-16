import os
import datetime
import pandas as pd
import pytorch_lightning as pl
from typing import OrderedDict, Tuple
# from transformers import AdamW
from torch.optim import AdamW
from src.model.model import *
from src.model.utils import *
from src.utils.lr_scheduler import *
from src.score.rouge_score import RougeScorer


class MatchSum_Engine(pl.LightningModule):

    __doc__ = r"""
        pl-based engine for training a extractive summarization model.
        Unlike the english benchmark datasets(CNN/DM etc.), it has human-written extractive labels,
        so we evaluate the model with the given extractive labels instead of the abstractive. 
    
        Args:
            model: model instance to train
            train_df: train dataset in pd.DataFrame
            val_df: validation dataset in pd.DataFrame
            test_df: test dataset in pd.DataFrame
            sum_size: # sentences in a model-predicted summary
            n_block: n-gram size for n-gram blocking 
            model_checkpoint: checkpoints of only model
            freeze_base: freeze the model parameters while training
            lr: learning rate
            betas: betas of torch.optim.Adam
            weight_decay: weight_decay of torch.optim.Adam
            adam_epsilon: eps of torch.optim.Adam
            num_warmup_steps: # warm-up steps 
            num_training_steps: # total training steps
            save_result: save test result
             
        train_df, val_df and test_df must be given in order to get the candidate summary 
        from the prediction by indexing the document.
    """

    def __init__(
            self,
            model,
            train_df: Optional[pd.DataFrame] = None,
            val_df: Optional[pd.DataFrame] = None,
            test_df: Optional[pd.DataFrame] = None,
            inference_df: Optional[pd.DataFrame] = None,
            sum_size: Optional[int] = 3,
            n_block: int = 3,
            model_checkpoint: Optional[str] = None,
            freeze_base: bool = False,
            lr: float = None,
            betas: Tuple[float] = (0.9, 0.999),
            weight_decay: float = 0.01,
            adam_epsilon: float = 1e-8,
            num_warmup_steps: int = None,
            num_training_steps: int = None,
            save_result: bool = False,
    ):
        super().__init__()

        self.model = model
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.sum_size = sum_size
        self.n_block = n_block

        # hparmas
        self.model_checkpoint = model_checkpoint
        self.freeze_base = freeze_base
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.save_result = save_result

        self.scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        self.prepare_training()


    def prepare_training(self):
        self.model.train()

        if self.model_checkpoint: # 모델 체크포인트 불러오기
            checkpoint = torch.load(self.model_checkpoint)
            assert isinstance(checkpoint, OrderedDict), 'please load lightning-format checkpoints'
            assert next(iter(checkpoint)).split('.')[0] != 'model', 'this is only for loading the model checkpoints'
            self.model.load_state_dict(checkpoint)

        if self.freeze_base: # base model의 파라미터는 학습 제외
            for p in self.model.base_model.parameters():
                p.requires_grad = False


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight'] # weight decay 없이 학습

        optim_params = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optim_params, self.lr, betas=self.betas, eps=self.adam_epsilon)
        scheduler = get_transformer_scheduler(optimizer, self.num_warmup_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {'scheduler': scheduler, 'interval': 'step'}
        }


    def on_train_epoch_start(self):
        self._train_outputs = []


    def training_step(self, batch, batch_idx):
        outputs = self.model(
            batch['encodings'],
            batch['cls_token_ids'],
            batch['ext_label'],
        )
        loss = outputs['loss']
        self.log('train_step_loss', loss, prog_bar=True)
        self._train_outputs.append(loss)

        return {'loss': loss} 


    def on_train_epoch_end(self):
        loss = torch.stack(self._train_outputs).mean()
        self.log('train_loss', loss, prog_bar=True)
        self._train_outputs = [] # reset

    
    def on_validation_epoch_start(self):
        self._val_outputs = []


    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            batch['encodings'],
            batch['cls_token_ids'],
            batch['ext_label'],
        )
        loss = outputs['loss']
        preds = outputs['prediction']

        ref_sums, can_sums = [], []
        accs = []
        for i, id in enumerate(batch['id']):
            sample = self.val_df[self.val_df['id'] == id].squeeze()
            text = sample['text']

            ref_sum = [text[i] for i in sample['extractive']]
            ref_sums.append('\n'.join(ref_sum))

            can_sum = get_candidate_sum(text, preds[i], self.sum_size, self.n_block)
            can_sums.append('\n'.join(can_sum))

            # accuracy 계산: acc = 정답 중 맞힌 개수 / 정답 문장 수
            ref_indices = set(sample['extractive']) # 정답 인덱스
            pred_indices = set(preds[i][:self.sum_size])

            if len(ref_indices) > 0:
                acc = len(ref_indices & pred_indices) / len(ref_indices)
            else:
                acc = 0.0
            accs.append(acc)

        output = {
            'loss': loss,
            'ref_sums': ref_sums,
            'can_sums': can_sums,
            'accs': accs
        }
        self._val_outputs.append(output)

        return output


    def on_validation_epoch_end(self):
        losses = []
        r1, r2, rL = [], [], []
        accs = []

        print('calculating ROUGE score & ACC...')
        for output in self._val_outputs:
            ref_sums = output['ref_sums']
            can_sums = output['can_sums']
            for ref_sum, can_sum in zip(ref_sums, can_sums):
                rouge = self.scorer.score(ref_sum, can_sum)
                r1.append(rouge['rouge1'].fmeasure)
                r2.append(rouge['rouge2'].fmeasure)
                rL.append(rouge['rougeL'].fmeasure)

            losses.append(output['loss'])
            accs.extend(output['accs'])

        loss = torch.stack(losses).mean()
        r1 = 100 * (sum(r1) / len(r1))
        r2 = 100 * (sum(r2) / len(r2))
        rL = 100 * (sum(rL) / len(rL))
        acc = 100 * (sum(accs) / len(accs))

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_rouge1', r1, prog_bar=True)
        self.log('val_rouge2', r2, prog_bar=True)
        self.log('val_rougeL', rL, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        self._val_outputs = [] # reset


    def on_test_epoch_start(self):
        self._test_outputs = []


    def test_step(self, batch, batch_idx):
        outputs = self.model(
            batch['encodings'],
            batch['cls_token_ids'],
        )
        preds = outputs['prediction']

        texts, ref_sums, can_sums, accs = [], [], [], []
        ref_idx, can_idx = [], []

        for i, id in enumerate(batch['id']):
            sample = self.test_df[self.test_df['id'] == id].squeeze()
            text = sample['text']
            texts.append('\n'.join(text))

            ref_sum = [text[i] for i in sample['extractive']]
            ref_sums.append('\n'.join(ref_sum))

            can_sum = get_candidate_sum(text, preds[i], self.sum_size, self.n_block)
            can_sums.append('\n'.join(can_sum))

            # accuracy 계산: acc = 정답 중 맞힌 개수 / 정답 문장 수
            ref_indices = set(sample['extractive']) # 정답 인덱스
            pred_indices = set(preds[i][:self.sum_size])
            ref_idx.append(ref_indices)
            can_idx.append(pred_indices)

            if len(ref_indices) > 0:
                acc = len(ref_indices & pred_indices) / len(ref_indices)
            else:
                acc = 0.0
            accs.append(acc)

        output = {
            'texts': texts,
            'ref_sums': ref_sums,
            'can_sums': can_sums,
            'ref_idx': ref_idx,
            'can_idx': can_idx,
            'accs': accs
        }
        self._test_outputs.append(output)

        return output


    def on_test_epoch_end(self):
        result = {
            'text': [],
            'reference summary': [],
            'candidate summary': [],
            'reference indices': [],
            'candidate indices': []
        }
        r1, r2, rL, accs = [], [], [], []

        print('calculating ROUGE score & ACC...')
        for output in self._test_outputs:
            texts = output['texts']
            ref_sums = output['ref_sums']
            can_sums = output['can_sums']

            result['reference indices'].append(output['ref_idx'])
            result['candidate indices'].append(output['can_idx'])

            for i, (ref_sum, can_sum) in enumerate(zip(ref_sums, can_sums)):
                rouge = self.scorer.score(ref_sum, can_sum)
                r1.append(rouge['rouge1'].fmeasure)
                r2.append(rouge['rouge2'].fmeasure)
                rL.append(rouge['rougeL'].fmeasure)

                if self.save_result:
                    result['text'].append(texts[i])
                    result['reference summary'].append(ref_sum)
                    result['candidate summary'].append(can_sum)

            accs.extend(output['accs'])

        r1 = 100 * (sum(r1) / len(r1))
        r2 = 100 * (sum(r2) / len(r2))
        rL = 100 * (sum(rL) / len(rL))
        acc = 100 * (sum(accs) / len(accs))

        print('rouge1: ', r1)
        print('rouge2: ', r2)
        print('rougeL: ', rL)
        print('accuracy: ', acc)

        if self.save_result:
            path = './result/{}'.format(datetime.datetime.now().strftime('%y-%m-%d'))
            if not os.path.exists(path):
                os.makedirs(path)

            result_pd = pd.DataFrame(result)
            result_pd.to_csv(path + '/{}.csv'.format(datetime.datetime.now().strftime('%H-%M-%S')), index=False)

        self._test_outputs = [] # reset

    def on_predict_epoch_start(self):
        self._predict_outputs = []


    def predict_step(self, batch, batch_idx):
        outputs = self.model(
            batch['encodings'],
            batch['cls_token_ids'],
        )
        preds = outputs['prediction']

        texts, ref_sums, can_sums, accs = [], [], [], []
        ref_idx, can_idx = [], []

        for i, id in enumerate(batch['id']):
            sample = self.inference_df[self.inference_df['id'] == id].squeeze()
            text = sample['text']
            texts.append('\n'.join(text))

            can_sum = get_candidate_sum(text, preds[i], self.sum_size, self.n_block)
            can_sums.append('\n'.join(can_sum))

            pred_indices = set(preds[i][:self.sum_size])
            can_idx.append(pred_indices)

        output = {
            'texts': texts,
            'can_sums': can_sums,
            'can_idx': can_idx,
        }
        self._predict_outputs.append(output)

        return output


    def on_predict_epoch_end(self):
        result = {
            'text': [],
            'candidate summary': [],
            'candidate indices': []
        }

        for output in self._test_outputs:
            texts = output['texts']
            can_sums = output['can_sums']

            result['candidate indices'].append(output['can_idx'])

            for i, can_sum in enumerate(can_sums):

                if self.save_result:
                    result['text'].append(texts[i])
                    result['candidate summary'].append(can_sum)

        if self.save_result:
            path = './result/{}'.format(datetime.datetime.now().strftime('%y-%m-%d'))
            if not os.path.exists(path):
                os.makedirs(path)

            result_pd = pd.DataFrame(result)
            result_pd.to_csv(path + '/{}.csv'.format(datetime.datetime.now().strftime('%H-%M-%S')), index=False)

        self._test_outputs = [] # reset