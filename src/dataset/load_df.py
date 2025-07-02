import json
import os
import sys
import pandas as pd
import numpy as np
import torch.nn as nn
from itertools import combinations
from typing import Optional, List
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.preprocess.ext_scorer import get_salience_score, load_scorer_model
from transformers import AutoTokenizer
from tqdm import tqdm

def get_df(path: str) -> pd.DataFrame:
    """
        json 파일에서 데이터를 불러와 DataFrame으로 변환
        Unlike the english benchmark datasets(CNN/DM etc.), it has human-written extractive labels,
        so no oracle algorithm is required.
    """

    with open(path, 'r') as f:
        data = json.load(f)
    
    data = data['documents']

    def prep_text(text):
        lines = []
        for paragraph in text:
            for line in paragraph:
                lines.append(line['sentence'])
        return lines

    new_data = []
    for doc in data:
        text = prep_text(doc['text'])
        s = {
            'id': doc['id'],
            'title': doc['title'],
            'text': text,
            'extractive': doc['extractive'],
            'abstractive': doc['abstractive']
        }
        if None in s['extractive']:
            continue
        new_data.append(s)
    
    return pd.DataFrame(new_data)

def flatten_candidate_df(
    df: pd.DataFrame, 
    model: nn.Module,
    tokenizer,
    candidate_num: int,
    sel_sent: int=3,
    topk_k: int=10,
    device: str='cuda',
    ):
    
    rows = []
    model.eval()
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        doc_id = row['id']
        text_sents = row['text']
        ext_indices = row['extractive']

        try:
            scores = get_salience_score(model, tokenizer, text_sents, device=device)
            scores = np.array(scores)
        except Exception as e:
            print(f'[ERROR] Scoring Failed for {doc_id}: {e}')
            continue

        topk_idx = scores.argsort()[::-1][:topk_k]
        candidate_indices = list(combinations(topk_idx, sel_sent))[:candidate_num]

        gold_summary = [text_sents[i] for i in ext_indices]

        candidate_list = []
        candidate_id_list = []
        label_idx = -1

        for cand_id, idx_tuple in enumerate(candidate_indices):
            candidate = [text_sents[i] for i in idx_tuple]
            candidate_list.append(candidate)
            candidate_id_list.append(cand_id)
            
            if set(idx_tuple) == set(ext_indices):
                label_idx = cand_id

        if label_idx == -1 or len(candidate_list) < 2:
            continue

        rows.append({
            'doc_id': doc_id,
            'candidate_id': candidate_id_list,
            'text': text_sents,
            'candidates': candidate_list,
            'gold_summary': gold_summary,
            'extractive': ext_indices,
            'label': label_idx
        })
        
    print('doc_id: ', rows[0]['doc_id'])
    print('cand_id: ', rows[0]['candidate_id'])
    print('extractive: ', rows[0]['extractive'])
    print('label: ', rows[0]['label'])
        
    return pd.DataFrame(rows)

def get_train_df(
        path: str,
        use_df: List[int] = [0, 1, 2], #0: 법률, 1: 사설, 2: 신문기사,
        model: nn.Module = None,
        tokenizer=None,
        candidate_num: Optional[int] = 20,
        val_ratio: float = 0.1,
        random_state: Optional[int] = 42,
        shuffle: bool = True,
) -> pd.DataFrame:
    
    print('Getting df_1...')
    df1 = get_df(os.path.join(path, 'data/Training/법률_train_original.json'))
    print('Got df_1.')
    print('Getting df_2')
    df2 = get_df(os.path.join(path, 'data/Training/사설_train_original.json'))
    print('Got df_2.')
    print('Getting df_3')
    df3 = get_df(os.path.join(path, 'data/Training/신문기사_train_original.json'))
    print('Got df_3.')

    df_list = [df1, df2, df3]
    use_df_list = [df_list[i] for i in use_df]

    print('Cancatenating the dfs by use_df...')

    df = pd.concat(use_df_list, ignore_index=True)
    df = df.dropna()
    df = df.drop_duplicates(subset=['id'], ignore_index=True)

    print('Concatenating the dfs by use_df completed.')

    print('Splitting df by train_test_split...')

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=random_state,
        shuffle=shuffle
    )
    
    print('Splitting df by train_test_split completed.')

    print('Getting flatten_candidate_df of train_df...')
    train_df = flatten_candidate_df(train_df, model, tokenizer, candidate_num)
    print('Got flatten_candidate_df of train_df.')
    print('Getting flatten_candidate_df of val_df...')
    val_df = flatten_candidate_df(val_df, model, tokenizer, candidate_num)
    print('Got flatten_candidate_df of val_df.')

    return train_df, val_df

def get_test_df(
        path: str,
        use_df: List[int] = [0, 1, 2]  # 0: 법률, 1: 사설, 2: 신문기사
):
    df1 = get_df(os.path.join(path, 'Validation/법률_valid_original.json'))
    df2 = get_df(os.path.join(path, 'Validation/사설_valid_original.json'))
    df3 = get_df(os.path.join(path, 'Validation/신문기사_valid_original.json'))

    df_list = [df1, df2, df3]
    use_df_list = [df_list[i] for i in use_df]

    df = pd.concat(use_df_list, ignore_index=True)
    df = df.dropna()
    df = df.drop_duplicates(subset=['id'], ignore_index=True)

    return df

if __name__ == '__main__':
    print('Importing Tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    print('Importing Tokenizer completed.')
    print('Importing Encoder Model...')
    encoder = load_scorer_model(
        tokenizer='klue/bert-base', 
        encoder_path='/home/food/people/hyeonjin/FoodSafetyCrawler/matchsum/BertExt/bertext_only.pt',
        enc_num_layers=2,
        enc_intermediate_size=2048,
        enc_num_attention_heads=8,
        enc_dropout_prob=0.1,
        device='cuda'
        )
    print('Importing Encoder Model completed.')
    
    print('Getting train_df and val_df...')
    train_df, val_df = get_train_df(
        path='/home/food/people/hyeonjin/FoodSafetyCrawler/aihub',
        use_df=[0], #0: 법률, 1: 사설, 2: 신문기사,
        model=encoder,
        tokenizer=tokenizer,
        candidate_num=20,
        val_ratio= 0.1,
        random_state = 42,
        shuffle = True,
    )
    print('Got train_df and val_df.')
    #print(train_df.head())
    #print(valid_df.head())

    for col in train_df.columns:
        print(f'{col}: ', train_df[col][0])
        
    print(len(train_df['text'][0]))

    for cand in train_df['candidates'][0]:
        print(cand)
        
    print(train_df)
    
    for _, row in train_df.iterrows():
        if row['label'] == -1:
            print(row['doc_id'])
    
    train_df.to_csv('/home/food/people/hyeonjin/FoodSafetyCrawler/matchsum/MatchSumRevised/data/train_df.csv', index=False)
    val_df.to_csv('/home/food/people/hyeonjin/FoodSafetyCrawler/matchsum/MatchSumRevised/data/val_df.csv', index=False)