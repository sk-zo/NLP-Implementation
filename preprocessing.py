import re
from tqdm import tqdm
import pickle
from datasets import load_dataset
from transformers import BertTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize


def get_ngram(n, words):
    ngram_set = set()
    n_words = len(words)
    max_start_index = n_words - n
    for i in range(max_start_index + 1):
        ngram_set.add(tuple(words[i: i+n]))
    return ngram_set

def cal_rouge(src, ref):
    n_src = len(src)
    n_ref = len(ref)
    n_overlap = len(src.intersection(ref))

    if n_overlap == 0:
        p = 0.0
    else:
        p = n_overlap / n_src

    if n_ref == 0:
        r = 0.0
    else:
        r = n_overlap / n_ref

    f1_score = 2.0 * ((p * r) / (p + r + 1e-6))

    return f1_score

if __name__ == '__main__':
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    print("dataset loaded")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("tokenizer loaded")


    preprocessed_dataset = []


    for row in tqdm(dataset['train']):
        _src = []
        src = row['article']
        src_sents = sent_tokenize(src)
        token_len = 0
        src_input_ids = []

        for sent in src_sents:
            cleand_sent = re.sub(r'[^a-zA-Z0-9 ]', '', sent)
            tokens = tokenizer(sent).input_ids
            if len(tokens) + token_len > 512:
                break
            src_input_ids.extend(tokens)
            token_len += len(tokens)
            _src.append(word_tokenize(cleand_sent))
            
        src_mask = [1] * token_len + [0] * (512-token_len)
        src_seg = [0] * 512
        src_tokens = {'input_ids':src_input_ids, 'token_type_ids':src_seg, 'attention_mask':src_mask}

        tgt = row['highlights']
        cleand_tgt = re.sub(r'[^a-zA-Z0-9 ]', '', tgt)
        _tgt = word_tokenize(cleand_tgt)
        tgt_tokens = tokenizer(cleand_tgt, max_length=512, padding="max_length")

        tgt_ngram_1 = get_ngram(1, _tgt)
        tgt_ngram_2 = get_ngram(2, _tgt)

        max_id = -1
        max_rouge = 0.0

        for i, _row in enumerate(_src):
            src_ngram_1 = get_ngram(1, _row)
            src_ngram_2 = get_ngram(2, _row)
            rouge_1 = cal_rouge(src_ngram_1, tgt_ngram_1)
            rouge_2 = cal_rouge(src_ngram_2, tgt_ngram_2)
            rouge_score = rouge_1 + rouge_2
            if rouge_score > max_rouge:
                max_rouge = rouge_score
                max_id = i

        clss = [i for i, token in enumerate(src_input_ids) if token == 101]
        tgt = clss[max_id]

        preprocessed_dataset.append({'src_tokens':src_tokens, 'clss':clss, 'tgt':tgt})

    with open('preprocessed_dataset', 'wb') as f:
        pickle.dump(preprocessed_dataset, f)
