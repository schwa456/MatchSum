import os
import datetime
import pandas as pd
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import BatchEncoding
from torch.optim import AdamW
from typing import Optional, Tuple, OrderedDict
from src.model.model import *
from src.model.utils import *
from src.utils.lr_scheduler import *
from src.preprocess.get_candidates import *
from src.score.rouge_score import RougeScorer


class MatchSum_Engine(pl.LightningModule):
    def __init__(self, 
        model,
        encoder,
        train_df: Optional[pd.DataFrame] = None,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        encoder_path: Optional[str] = None,
        sum_size: Optional[int] = 3,
        pred_sum_size: Optional[int] = 3,
        model_checkpoint: Optional[str] = None,
        freeze_base: bool = False,
        lr: float = None,
        betas: Tuple[float] = (0.9, 0.999),
        margin: float = 0.1,
        candidate_num: int = 10,
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
        self.pred_sum_size = pred_sum_size
        self.model_checkpoint = model_checkpoint
        
        self.freeze_base = freeze_base
        self.lr = lr
        self.betas = betas
        self.margin = margin
        self.candidate_num = candidate_num
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.save_result = save_result
        
        self.encoder = encoder
        self.loss_fn = nn.MarginRankingLoss(margin=self.margin)
        self.scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        
        self.prepare_training()

    def forward(self, doc_input_ids, doc_attention_mask, doc_cls_indices,
                cand_input_ids, cand_attention_mask, cand_cls_indices):
        """
        doc_input_ids: (B, L)
        cand_input_ids: (B, N, L)
        """
        # Encode document
        doc_encodings = BatchEncoding({
            'input_ids': doc_input_ids,
            'attention_mask': doc_attention_mask
        })
        doc_output = self.encoder(doc_encodings, doc_cls_indices)
        doc_embed = doc_output['cls_embeddings'].mean(dim=1)
        
        # shape = (B, L) → reshape to (B, 1, L)
        if cand_input_ids.dim() == 2:
            cand_input_ids = cand_input_ids.unsqueeze(1)
            cand_attention_mask = cand_attention_mask.unsqueeze(1)
            cand_cls_indices = cand_cls_indices.unsqueeze(1)
            
        # Encode each candidate (flatten batch * N)    
        B, N, L = cand_input_ids.size()
        cand_input_ids = cand_input_ids.view(B * N, L)
        cand_attention_mask = cand_attention_mask.view(B * N, L)
        cand_encodings = BatchEncoding({
            'input_ids': cand_input_ids.view(B * N, L),
            'attention_mask': cand_attention_mask.view(B * N, L)
        })
        cand_output = self.encoder(cand_encodings, cand_cls_indices.squeeze(1))
        cand_embed = cand_output['cls_embeddings'].mean(dim=1).view(B, N, -1)
        
        return doc_embed, cand_embed

    def prepare_training(self):
        self.model.train()
        
        if self.model_checkpoint: # Loading Model Checkpoint
            checkpoint = torch.load(self.model_checkpoint)
            assert isinstance(checkpoint, OrderedDict), 'Please Load Lightning-Format Checkpoints'
            assert next(iter(checkpoint)).split('.')[0] != 'model', 'This is Only For Loading The Model Checkpoints'
            self.model.load_state_dict(checkpoint)
        
        if self.freeze_base: # Base Model의 파라미터는 학습 제외
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
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
        }

    def on_train_epoch_start(self):
        self._train_outputs = []
        
    def training_step(self, batch, batch_idx):
        doc_input_ids = batch["doc_input_ids"]     # (B, L)
        doc_attention_mask = batch["doc_attention_mask"]
        doc_cls_indices = batch['doc_cls_indices']
        cand_input_ids = batch["cand_input_ids"]   # (B, N, L)
        cand_attention_mask = batch["cand_attention_mask"]
        cand_cls_indices = batch['cand_cls_indices']
        label = batch["label"]                   # (B, N)

        doc_embed, cand_embed = self.forward(
            doc_input_ids=doc_input_ids,
            doc_attention_mask=doc_attention_mask,
            doc_cls_indices=doc_cls_indices,
            cand_input_ids=cand_input_ids,
            cand_attention_mask=cand_attention_mask,
            cand_cls_indices=cand_cls_indices
        )  # doc_embed: (B, H), cand_embed: (B, N, H)

        print(f"[DEBUG] label: {label}")                      # (B,) 또는 (B, N)
        print(f"[DEBUG] cand_input_ids.shape: {cand_input_ids.shape}")  # (B, N, L)
        print(f"[DEBUG] cand_embed.shape: {cand_embed.shape}")          # (B, N, H)

        # Compute cosine similarity
        sims = F.cosine_similarity(
            doc_embed.unsqueeze(1),  # (B, 1, H)
            cand_embed,              # (B, N, H)
            dim=-1
        )  # (B, N)

        # Positive: label이 가장 높은 candidate
        if label.dim() == 1:
            pos_idx = label  # Already (B,)
        elif label.dim() == 2:
            pos_idx = label.argmax(dim=1)
        else:
            raise ValueError(f'label dim expected to be 1 or 2, but got {label.dim()}')
        
        # Index 범위 체크
        B, N = sims.shape
        if torch.any(pos_idx >= N):
            raise ValueError(f'[ERROR] pos_idx out of range: max={pos_idx.max().item()} vs sims.shape={sims.shape}')
        
        pos_score = sims[torch.arange(B), pos_idx]  # (B,)
        
        # Negative: 그 외에서 가장 유사한 것 하나만 사용
        sims_clone = sims.clone()
        sims_clone[torch.arange(B), pos_idx] = -1e9
        neg_score = sims_clone.max(dim=1).values  # (B,)

        # Margin Ranking Loss
        loss = self.loss_fn(
            pos_score, neg_score,
            torch.ones_like(pos_score)
        )
        
        if loss is None:
            raise ValueError("Loss is None — check forward logic.")
        
        print("sims:", sims)
        print("pos_score:", pos_score)
        print("neg_score:", neg_score)
        print("loss:", loss)
        print("label:", label)
        print("cand_input_ids.shape:", cand_input_ids.shape)
        
        print("label (raw):", label)
        print("pos_idx:", pos_idx)
        print("pos_idx.max():", pos_idx.max().item())
        print("sims.shape:", sims.shape)

        self.log("train_step_loss", loss, prog_bar=True, on_step=True)
        return loss

    def on_train_epoch_end(self):
        loss = torch.stack(self._train_outputs).mean()
        self.log('train_loss', loss, prog_bar=True)
        self._train_outputs.clear()
    
    def on_validation_epoch_start(self):
        self._val_outputs = []
        
    def validation_step(self, batch, batch_idx):
        doc_input_ids = batch["doc_input_ids"]     # (B, L)
        doc_attention_mask = batch["doc_attention_mask"]
        cand_input_ids = batch["cand_input_ids"]   # (B, N, L)
        cand_attention_mask = batch["cand_attention_mask"]
        labels = batch["label"]                   # (B, N)

        doc_embed, cand_embed = self.forward(
            doc_input_ids=doc_input_ids,
            doc_attention_mask=doc_attention_mask,
            cand_input_ids=cand_input_ids,
            cand_attention_mask=cand_attention_mask
        )  # doc_embed: (B, H), cand_embed: (B, N, H)

        # Compute cosine similarity
        sims = F.cosine_similarity(
            doc_embed.unsqueeze(1),  # (B, 1, H)
            cand_embed,              # (B, N, H)
            dim=-1
        )  # (B, N)

        # Positive: label이 가장 높은 candidate
        pos_idx = labels.argmax(dim=1) # (B, )
        pos_score = sims.gather(1, pos_idx.unsqueeze(1)).squeeze(1)  # (B,)
        
        # Negative: 그 외에서 가장 유사한 것 하나만 사용
        sims_clone = sims.clone()
        sims_clone[torch.arange(len(sims)), pos_idx] = -1e9
        neg_score = sims_clone.max(dim=1).values  # (B,)

        # Margin Ranking Loss
        loss = self.loss_fn(
            pos_score, neg_score,
            torch.ones_like(pos_score)
        )

        self.log("valid_step_loss", loss, prog_bar=True, on_step=True)
        return loss

    def on_validation_epoch_end(self):
        losses = []
        r1, r2, rL = [], [], []
        accs = []
        
        print('Calculating ROUGE Score & ACC...')
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
        
        print("VAL collected losses:", len(losses))
        if losses:
            avg_loss = torch.stack(losses).mean()
        else:
            avg_loss = torch.tensor(0.0, device=self.device)
        
        r1 = 100 * (sum(r1) / len(r1))
        r2 = 100 * (sum(r2) / len(r2))
        rL = 100 * (sum(rL) / len(rL))
        acc = 100 * (sum(accs) / len(accs))
        
        self.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)
        self.log('val_rouge1', r1, prog_bar=True, sync_dist=True)
        self.log('val_rouge2', r2, prog_bar=True, sync_dist=True)
        self.log('val_rougeL', rL, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        
        self._val_outputs.clear()
    
    def on_test_epoch_start(self):
        self._test_outputs = []


    def test_step(self, batch, batch_idx):
        doc_input_ids = batch["doc_input_ids"]     # (B, L)
        doc_attention_mask = batch["doc_attention_mask"]
        cand_input_ids = batch["cand_input_ids"]   # (B, N, L)
        cand_attention_mask = batch["cand_attention_mask"]
        labels = batch["label"]                   # (B, N)

        doc_embed, cand_embed = self.forward(
            doc_input_ids=doc_input_ids,
            doc_attention_mask=doc_attention_mask,
            cand_input_ids=cand_input_ids,
            cand_attention_mask=cand_attention_mask
        )  # doc_embed: (B, H), cand_embed: (B, N, H)

        # Compute cosine similarity
        sims = F.cosine_similarity(
            doc_embed.unsqueeze(1),  # (B, 1, H)
            cand_embed,              # (B, N, H)
            dim=-1
        )  # (B, N)

        # Positive: label이 가장 높은 candidate
        pos_idx = labels.argmax(dim=1) # (B, )
        pos_score = sims.gather(1, pos_idx.unsqueeze(1)).squeeze(1)  # (B,)
        
        # Negative: 그 외에서 가장 유사한 것 하나만 사용
        sims_clone = sims.clone()
        sims_clone[torch.arange(len(sims)), pos_idx] = -1e9
        neg_score = sims_clone.max(dim=1).values  # (B,)

        # Margin Ranking Loss
        loss = self.loss_fn(
            pos_score, neg_score,
            torch.ones_like(pos_score)
        )

        self.log("test_step_loss", loss, prog_bar=True, on_step=True)
        return loss


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

        ids, texts, can_sums, can_idx = [], [], [], []

        for i, id in enumerate(batch['id']):
            ids.append(id)
            sample = self.inference_df[self.inference_df['id'] == id].squeeze()
            text = sample['text']
            texts.append('\n'.join(text))

            can_sum = get_candidate_sum(text, preds[i], self.pred_sum_size)
            can_sums.append('\n'.join(can_sum))

            pred_indices = set(preds[i][:self.pred_sum_size])
            can_idx.append(pred_indices)

        output = {
            'ids': ids,
            'texts': texts,
            'can_sums': can_sums,
            'can_idx': can_idx,
        }
        self._predict_outputs.append(output)

        return output


    def on_predict_epoch_end(self):
        result = {
            'ids': [],
            'text': [],
            'candidate summary': [],
            'candidate indices': []
        }

        for output in self._predict_outputs:
            ids = output['ids']
            texts = output['texts']
            can_sums = output['can_sums']
            can_idx = output['can_idx']

            result['ids'].extend(ids)
            result['text'].extend(texts)
            result['candidate summary'].extend(can_sums)
            result['candidate indices'].extend(can_idx)

        if self.save_result:
            path = './result/{}'.format(datetime.datetime.now().strftime('%y-%m-%d'))
            if not os.path.exists(path):
                os.makedirs(path)

            result_pd = pd.DataFrame(result)
            result_pd.to_csv(path + '/{}.csv'.format(datetime.datetime.now().strftime('%H-%M-%S')), index=False)

        self._predict_outputs = [] # reset