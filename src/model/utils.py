from typing import Optional, List

def get_ngrams(
        tokens: List[str],
        n: int
):
    """
        Generate N-gram Set from given Token List
    """

    ngrams = set()
    num = len(tokens) - n
    for i in range(num + 1):
        ngrams.add(tuple(tokens[i:i + n]))
    return ngrams

def ngram_blocking(
        sent: str,  # 새로 고려 중인 문장
        can_sum: List[str], # 현재까지 선택된 요약 문장 리스트
        ngram: int # 중복을 차단할 n-gram 크기
):
    """
        Checking whether given sentence is duplicated with selected sentences
    """

    sent_tri = get_ngrams(sent.split(), ngram)
    for can_sent in can_sum:
        can_tri = get_ngrams(can_sent.split(), ngram)
        if len(sent_tri.intersection(can_tri)) > 0:
            return True # 중복 있으면 True
    return False # 중복 없으면 False

def tri_blocking(sent, can_sum):
    return ngram_blocking(sent, can_sum, 3)

def quad_blocking(sent, can_sum):
    return ngram_blocking(sent, can_sum, 4)

def get_candidate_sum(
        text: str,
        prediction: List[int],
        sum_size: Optional[int] = None,
):
    """
        주어진 Text와 예측된 문장 Index List를 기반으로 요약 후보 문장 생성
        중복되는 n-gram이 있는 문장 제외
    """

    can_sum = []
    for i, sent_id in enumerate(prediction):
        if sent_id >= len(text):
            continue
        sent = text[sent_id]
        can_sum.append(sent)
        
        # 최대 문장 수 제한 도달 시 중단
        if sum_size and (len(can_sum) == sum_size):
            break
    
    return can_sum