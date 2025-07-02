import torch
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Dict
from .metric import *

class MatchSum(nn.Module):

    __doc__ = r"""
        Implementation of the paper;
        https://arxiv.org/abs/2004.08795
    """
    
    def __init__(
            self,
            candidate_num: Optional[int],
            tokenizer: Optional[str],
            margin: Optional[float],
            hidden_size: int = 768
    ):
        super(MatchSum, self).__init__()

        self.hidden_size = hidden_size
        self.candidate_num = candidate_num

        self.encoder = AutoModel.from_pretrained(tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.loss_fn = MarginRankingLoss(margin=margin)

            # Check whether Model Params are frozen
        for name, param in self.named_parameters():
            if not param.requires_grad:
                print("[Frozen]", name)


    def forward(self, 
                text_id: Dict[str, torch.Tensor], 
                candidate_id: Dict[str, torch.Tensor], 
                summary_id: Dict[str, torch.Tensor]):
        
        batch_size, candidate_num, seq_len = candidate_id['input_ids'].size()

        # get document embedding
        doc_out = self.encoder(
            input_ids = text_id['input_ids'], 
            attention_mask = text_id['attention_mask']
        )[0] # last layer
        doc_emb = doc_out[:, 0, :]
        assert doc_emb.size() == (batch_size, self.hidden_size) # [batch_size, hidden_size]
        
        # get summary embedding
        sum_out = self.encoder(
            input_ids=summary_id['input_ids'],
            attention_mask=summary_id['attention_mask']
        )[0] # last layer
        sum_emb = sum_out[:, 0, :]
        assert sum_emb.size() == (batch_size, self.hidden_size) # [batch_size, hidden_size]

        candidate_input_ids = candidate_id['input_ids'].view(-1, seq_len)
        candidate_attention = candidate_id['attention_mask'].view(-1, seq_len)

        # get candidate embedding
        cand_out = self.encoder(
            input_ids=candidate_input_ids,
            attention_mask=candidate_attention
        )[0]
        candidate_emb = cand_out[:, 0, :].view(batch_size, candidate_num, self.hidden_size)  # [batch_size, candidate_num, hidden_size]
        assert candidate_emb.size() == (batch_size, candidate_num, self.hidden_size)

        # get summary score
        summary_score = torch.cosine_similarity(sum_emb, doc_emb, dim=-1)
        
        # get candidate score
        doc_exp = doc_emb.unsqueeze(1).expand_as(candidate_emb)
        cand_score = torch.cosine_similarity(candidate_emb, doc_exp, dim=-1) # [batch_size, candidate_num]
        assert cand_score.size() == (batch_size, candidate_num)
        pred_idx = torch.argmax(cand_score, dim=1)

        loss = None
        if summary_id is not None:
            loss = self.loss_fn(score=cand_score, summary_score=summary_score)

        return {
            'score': cand_score, 
            'summary_score': summary_score,
            'prediction': pred_idx,
            'loss': loss
        }

