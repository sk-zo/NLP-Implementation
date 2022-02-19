import torch.utils.data as data
from tqdm import tqdm


class BertDataset(data.Dataset):
    def __init__(self, dataset, tokenizer, sent, label, max_len, mode='train'):
        self.mode = mode
        self.len = len(dataset)
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.labels = []
        if mode =='train':
            for row in tqdm(dataset, total=len(dataset)):
                tokens = tokenizer(row[sent], max_length=max_len, padding="max_length", truncation=True, add_special_tokens=False, return_tensors='pt')
                self.input_ids.append(tokens['input_ids'].squeeze(0))
                self.token_type_ids.append(tokens['token_type_ids'].squeeze(0))
                self.attention_mask.append(tokens['attention_mask'].squeeze(0))
                self.labels.append(row[label])
        else:
            for row in dataset:
                tokens = tokenizer(row[sent], max_length=max_len, padding="max_length", truncation=True, add_special_tokens=False, return_tensors='pt')
                self.input_ids.append(tokens['input_ids'].squeeze(0))
                self.token_type_ids.append(tokens['token_type_ids'].squeeze(0))
                self.attention_mask.append(tokens['attention_mask'].squeeze(0))

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.labels[index]
        else:
            return (self.input_ids[index], self.token_type_ids[index], self.attention_mask[index])
    
    def __len__(self):
        return self.len

class BartDataset(data.Dataset):
    def __init__(self, dataset, tokenizer, input, ref, max_len, mode='train'):
        self.mode = mode
        self.len = len(dataset)
        self.input_ids = []
        self.attention_mask = []
        self.decoder_input_ids = []
        self.decoder_attention_mask = []
        self.references=  []
        if mode =='train':
            for row in tqdm(dataset, total=len(dataset)):
                input_tokens = tokenizer(row[input], max_length=max_len, padding="max_length", add_special_tokens=True, truncation=True, return_tensors='pt')
                self.input_ids.append(input_tokens['input_ids'].squeeze(0))
                self.attention_mask.append(input_tokens['attention_mask'].squeeze(0))

                decoder_input_tokens = tokenizer(row[ref], max_length=max_len, padding="max_length", add_special_tokens=True, truncation=True, return_tensors='pt')
                self.decoder_input_ids.append(decoder_input_tokens['input_ids'].squeeze(0))
                self.decoder_attention_mask.append(decoder_input_tokens['attention_mask'].squeeze(0))
        else:
            for row in dataset:
                input_tokens = tokenizer(row[input], max_length=max_len, padding="max_length", truncation=True, return_tensors='pt')
                self.input_ids.append(input_tokens['input_ids'].squeeze(0))
                self.attention_mask.append(input_tokens['attention_mask'].squeeze(0))
                self.references.append([row[ref]])

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.input_ids[index], self.attention_mask[index], self.decoder_input_ids[index], self.decoder_attention_mask[index]
        else:
            return self.input_ids[index], self.attention_mask[index], self.references[index]
    
    def __len__(self):
        return self.len
    
