import math
import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer

class MatchSumEncoder(nn.Module):
    """
    MatchSumEncoder is a custom encoder for the MatchSum model.
    """
    __doc__ = r"""
        cls_attention_mask prevents padding tokens from being included in softmax values 
        inside the encoder's self-attention layer.
        (cls_attention_mask: CLS 토큰이 padding 위치에 있을 경우, self-attention에서 softmax 값에 포함되지 않도록 방지)
        """
    def __init__(
            self,
            num_layers: int, # BERT 인코더 레이어 수
            hidden_size: int, # BERT 인코더의 임베딩 차원
            intermediate_size: int, # FFN의 중간 레이어 차원
            num_attention_heads: int, # self-attention의 헤드 수
            dropout_prob: float, # dropout 확률
    ):

        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.dropout_prob = dropout_prob

        # position embedding 정의
        self.position_embedding = PositionEmbedding(dropout_prob, hidden_size)
        # BERT 인코더 레이어를 정의하여 ModuleList에 저장
        self.layers = nn.ModuleList([self.bert_layer() for _ in range(self.num_layers)])

        # 모델의 마지막 출력층
        # 각 문장(CLS 임베딩)에 대해 추출 요약 문장인지를 1개의 스코어(logit)로 예측
        self.last_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, eps=1e-6),
            nn.Linear(hidden_size, 1, bias=True)
        )

    def bert_layer(self):
        # BERT 인코더 레이어 하나를 생성하는 함수
        config = BertConfig()
        config.hidden_size = self.hidden_size
        config.intermediate_size = self.intermediate_size
        config.num_attention_heads = self.num_attention_heads
        config.hidden_dropout_prob = self.dropout_prob

        return BertLayer(config)

    def cls_attention_mask(self, cls_token_mask):
        # cls_token_mask를 attention에서 사용할 수 있도록 변환
        # shape: [batch, num_head, seq_len, seq_len]

        attention_mask = cls_token_mask[:, None, None, : ]# (batch, 1, 1, seq_len)
        attention_mask = attention_mask.expand(
            -1, self.num_attention_heads, attention_mask.size(-1), -1)  # (batch, num_head, seq_len, seq_len)
        attention_mask = (1.0 - attention_mask) * -1e18 # 마스킹된 위치는 매우 작은 값으로 설정(softmax에서 제외됨)

        return attention_mask

    def forward(self, last_hidden_state, cls_token_ids):
        # last_hidden_state: (batch, seq_len, hidden_dim) BERT의 마지막 출력
        # cls_token_ids: (batch, num_sentences) 각 문장의 CLS 토큰 위치 인덱스
        cls_token_mask = (cls_token_ids != -1).float() # 유효한 CLS 위치인지 확인(패딩은 -1) (batch, num_sentences)
        cls_index = torch.arange(last_hidden_state.size(0)).unsqueeze(1), cls_token_ids
        cls_embed = last_hidden_state[cls_index] # (batch, num_sentences, hidden_dim)
        cls_embed = cls_embed * cls_token_mask[:, :, None] # 패딩 CLS 무효화

        if self.num_layers:
            # 위치 임베딩 추가
            pos_embed = self.position_embedding.pe[:, :last_hidden_state.size(1)]
            cls_embed = cls_embed + pos_embed # 위치 정보 추가
            attention_mask = self.cls_attention_mask(cls_token_mask) # attention mask 생성

            # BERT 인코더 레이어를 순차적으로 통과
            for i in range(self.num_layers):
                cls_embed = self.layers[i](cls_embed, attention_mask=attention_mask)[0]
            cls_embed = cls_embed * cls_token_mask[:, :, None] # 패딩 CLS 무효화

        # 최종 스코어(logits) 출력
        logits = self.last_layer(cls_embed).squeeze(-1)
        logits = logits * cls_token_mask # 패딩 CLS 무효화

        return {
            'cls_embeddings': cls_embed, # 인코딩된 CLS 임베딩
            'cls_token_mask': cls_token_mask, # 유효 CLS 마스크
            'logits': logits, # 각 문장에 대한 요약 여부 스코어
        }

class PositionEmbedding(nn.Module):

    def __init__(
            self,
            dropout_prob: float,
            dim: int,
            max_len: int = 5000
    ):
        # 사인-코사인 기반 위치 임베딩 생성
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))

        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0) # batch 차원 추가 (1, max_len, dim)

        super().__init__()
        self.register_buffer('pe', pe) # 학습되지 않도록 buffer로 등록
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dim = dim


    def get_embed(self, embed):
        return self.pe[:, :embed.size(1)] # 주어진 임베딩 길이에 맞는 위치 임베딩 반환


    def forward(self, embed, step=None):
        embed = embed * math.sqrt(self.dim) # 임베딩 스케일링
        # step이 주어지면 해당 위치의 임베딩을 더하고, 아니면 전체 위치 임베딩을 더함
        # step은 현재 위치의 인덱스(예: 배치 내 문장 위치)
        if step:
            embed = embed + self.pe[:, step][:, None, :] # 특정 위치만 더할 경우
        else:
            embed = embed + self.pe[:, :embed.size(1)] # 전체 위치를 더할 경우
        embed = self.dropout(embed)
        return embed