import json
import os
import pandas as pd
from typing import Optional, List
from sklearn.model_selection import train_test_split

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
                line.append(line['sentence'])
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

def get_train_df(
        path: str,
        use_df: List[int] = [0, 1, 2], #0: 법률, 1: 사설, 2: 신문기사
        val_ratio: float = 0.1,
        random_state: Optional[int] = 42,
        shuffle: bool = True
) -> pd.DataFrame:
    df1 = get_df(os.parh.join(path, 'Training/법률_train_original.json'))
    df2 = get_df(os.path.join(path, 'Training/사설_train_original.json'))
    df3 = get_df(os.path.join(path, 'Training/신문기사_train_original.json'))

    df_list = [df1, df2, df3]
    use_df_list = [df_list[i] for i in use_df]

    df = pd.concat(use_df_list, ignore_index=True)
    df = df.dropna()
    df = df.drop_duplicates(subset=['id'], ignore_index=True)

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=random_state,
        shuffle=shuffle
    )

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