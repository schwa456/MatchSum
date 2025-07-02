import numpy as np
from os.path import join
import torch
import torch.nn as nn
from datetime import timedelta
from time import time

import evaluate

class MarginRankingLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, score, summary_score):
        # equivalent to initializing total_loss to 0
        # here is to avoid that some special samples will not go into the following for loop
        device = score.device
        total_loss = nn.MarginRankingLoss(0.0)(
            score, score, torch.ones_like(score).to(device))

        # candidate loss
        n = score.size(1)
        for i in range(1, n):
            pos_score = score[:, :-i].contiguous().view(-1)
            neg_score = score[:, i:].contiguous().view(-1)
            ones = torch.ones_like(pos_score).to(device)
            total_loss += nn.MarginRankingLoss(margin=self.margin * i)(pos_score, neg_score, ones)

        # gold summary loss
        pos_score = summary_score.unsqueeze(-1).expand_as(score).contiguous().view(-1)
        neg_score = score.contiguous().view(-1)
        ones = torch.ones_like(pos_score).to(device)

        total_loss += nn.MarginRankingLoss(margin=self.margin)(pos_score, neg_score, ones)

        return total_loss

class ValidMetric:
    def __init__(self, save_path, data):
        self.save_path = save_path
        self.data = data
        self.rouge = evaluate.load("rouge")
        self.reset()

    def reset(self):
        self.top1_correct = 0
        self.top6_correct = 0
        self.top10_correct = 0
        self.ROUGE_scores = []
        self.Error = 0
        self.cur_idx = 0

    def fast_rouge(self, dec, ref):
        result = self.rouge.compute(prediction=[dec], references=[ref])
        return (result['rouge1'] + result['rouge2'] + result['rougeL']) / 3.0

    def evaluate(self, score):
        batch_size = score.size(0)
        top_k = torch.max(score, dim=1).indices
        self.top1_correct += (top_k == 0).sum().item()
        self.top6_correct += (top_k < 6).sum().item()
        self.top10_correct += (top_k < 10).sum().item()

        for i in range(batch_size):
            max_idx = top_k[i].item()
            if max_idx >= len(self.data[self.cur_idx]['indices']):
                self.Error += 1
                self.cur_idx += 1
                continue
            ext_idx = sorted(self.data[self.cur_idx]['indices'][max_idx])
            dec = ' '.join([self.data[self.cur_idx]['text'][j] for j in ext_idx])
            ref = ' '.join(self.data[self.cur_idx]['summary'])
            self.ROUGE_scores.append(self.fast_rouge(dec, ref))
            self.cur_idx += 1

    def get_metric(self, reset=True):
        top1_accuracy = self.top1_correct / self.cur_idx
        top6_accuracy = self.top6_correct / self.cur_idx
        top10_accuracy = self.top10_correct / self.cur_idx
        ROUGE = np.mean(self.ROUGE_scores)
        result = {
            'top1_accuracy': top1_accuracy,
            'top6_accuracy': top6_accuracy,
            'top10_accuracy': top10_accuracy,
            'Error': self.Error,
            'ROUGE': ROUGE,
        }

        with open(join(self.save_path, 'train_info.txt'), 'a') as f:
            print(f'top1_accuracy: {top1_accuracy:.4f}, top6_accuracy: {top6_accuracy:.4f}, top10_accuracy: {top10_accuracy:.4f}, Error: {self.Error}, ROUGE: {ROUGE:.4f}', file=f)
        if reset:
            self.reset()
        return result

class MatchRougeMetric:
    def __init__(self, data, n_total):
        self.data = data
        self.n_total = n_total
        self.rouge = evaluate.load("rouge")
        self.reset()

    def reset(self):
        self.ext = []
        self.predictions = []
        self.references = []
        self.cur_idx = 0
        self.start = time()

    def evaluate(self, score):
        ext = int(torch.max(score, dim=1).indices)
        self.ext.append(ext)
        self.cur_idx += 1
        print(f'{self.cur_idx}/{self.n_total} ({self.cur_idx/self.n_total*100:.2f}%) decoded in {timedelta(seconds=int(time() - self.start))}\r', end='')

    def get_metric(self, reset=True):
        for i, ext in enumerate(self.ext):
            sent_ids = self.data[i]['indices'][ext]
            pred = ' '.join([self.data[i]['text'][j] for j in sent_ids])
            ref = ' '.join(self.data[i]['summary'])
            self.predictions.append(pred)
            self.references.append(ref)
        results = self.rouge.compute(predictions=self.predictions, references=self.references, use_stemmer=True)

        if reset:
            self.reset()

        return {
            'ROUGE-1': results['rouge1'],
            'ROUGE-2': results['rouge2'],
            'ROUGE-L': results['rougeL']
        }