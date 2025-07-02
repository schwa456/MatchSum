from torch.utils.data import Dataset
from collections import defaultdict
from transformers import PreTrainedTokenizer
import json, torch, ast

class MatchSumDataset(Dataset):
    def __init__(self, df, tokenizer: PreTrainedTokenizer, max_len=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        self.flat_rows = []
        for i, row in df.iterrows():
            text = row['text']
            doc_text = ''.join(text)
            label_idx = row['label']
            cands = ast.literal_eval(row['candidates'])
            for i, cand in enumerate(cands):
                self.flat_rows.append({
                    'doc': doc_text,
                    'cand': '\n'.join(cand),
                    'label': int(i == label_idx)
                })

    def __getitem__(self, idx):
        row = self.flat_rows[idx]
        print(f"\n[DEBUG] row: {row}")
        print(f"\n[DEBUG] doc: {row['doc']}, \ncandidate: {row['cand']}, \nlabel: {row['label']}")
        print(f"\n[DEBUG] type(doc): {type(row['doc'])}, \ncandidate: {type(row['cand'])}, \nlabel: {type(row['label'])}")
        doc = row['doc']
        cand = row['cand']
        label = row['label']
        
        doc_sents = doc.strip().split('\n')
        cand_sents = cand.strip().split('\n')
        
        def encode_with_cls_indices(sents, tokenizer, max_len):
            input_ids = []
            cls_indices = []
            for sent in sents:
                tokens = tokenizer.encode(sent, add_special_tokens=True)
                if len(input_ids) + len(tokens) > max_len:
                    break
                cls_indices.append(len(input_ids))
                input_ids.extend(tokens)
            attention_mask = [1] * len(input_ids)
            return input_ids, attention_mask, cls_indices
        
        doc_input_ids, doc_attention_mask, doc_cls_indices = encode_with_cls_indices(
            doc_sents, self.tokenizer, self.max_len
        )
        cand_input_ids, cand_attention_mask, cand_cls_indices = encode_with_cls_indices(
            cand_sents, self.tokenizer, self.max_len
        )
        
        def pad(toks, length):
            return toks + [0] * (length - len(toks))
        
        doc_input_ids = pad(doc_input_ids, self.max_len)
        doc_attention_mask = pad(doc_attention_mask, self.max_len)
        cand_input_ids = pad(cand_input_ids, self.max_len)
        cand_attention_mask = pad(cand_attention_mask, self.max_len)

        return {
            'doc_input_ids': torch.tensor(doc_input_ids),
            'doc_attention_mask': torch.tensor(doc_attention_mask),
            'doc_cls_indices': torch.tensor(doc_cls_indices),
            'cand_input_ids': torch.tensor(cand_input_ids),
            'cand_attention_mask': torch.tensor(cand_attention_mask),
            'cand_cls_indices': torch.tensor(cand_cls_indices),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.flat_rows)
    
if __name__ == '__main__':
    import pandas as pd
    from transformers import AutoTokenizer
    train_df = pd.read_csv('/home/food/people/hyeonjin/FoodSafetyCrawler/matchsum/MatchSumRevised/data/train_df.csv')
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    
    candidates = train_df.loc[0, 'candidates']
    candidates_list = ast.literal_eval(candidates)
    print(f"Number of candidates for doc 0: {len(candidates_list)}")
    
    train_dataset = MatchSumDataset(train_df, tokenizer)
    
    print(train_dataset[0])
    print(train_dataset[1])
    
    for i in range(10):
        print(train_dataset[i])
    
    print(len(train_df))