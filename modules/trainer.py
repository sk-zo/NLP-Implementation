import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import AdamW
from tqdm import tqdm
from datasets import load_metric

class Trainer:
    def __init__(self, args, model, tokenizer, dataset):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        
    def fit(self, eval=True):
        loss_fct = CrossEntropyLoss()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        device = self.args.device
        max_epoch = self.args.max_epoch
        max_grad_norm = self.args.max_grad_norm

        if self.args.task=='CL':
            for _ in range(max_epoch):
                self.model.train()
                for (input_ids, segment_ids, attention_mask, label) in tqdm(self.dataset['train'], total=len(self.dataset['train'])):
                    input_ids = input_ids.long().to(device)
                    segment_ids = segment_ids.long().to(device)
                    attention_mask = attention_mask.float().to(device)
                    label = label.long().to(device)
                    
                    out = self.model(token_ids=input_ids, segment_ids=segment_ids, attention_mask=attention_mask)
                    loss = loss_fct(out, label)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()
                print(f"epoch: {epoch+1}, loss: {loss.item()}")
            if eval==True:
                self.model.eval()
                total = 0
                for (input_ids, segment_ids, attention_mask, label) in self.dataset['valid']:
                    input_ids = input_ids.long().to(device)
                    segment_ids = segment_ids.long().to(device)
                    attention_mask = attention_mask.float().to(device)
                    label = label.long().to(device)
                    out = self.model(token_ids=input_ids, segment_ids=segment_ids, attention_mask=attention_mask)
                    max_vals, max_indices = torch.max(out, 1)
                    train_acc = (max_indices == label).sum().data.cpu().numpy()/max_indices.size()[0]
                    total+=train_acc
                print(total / len(self.dataset['valid']))

        elif self.args.task=='LM':
            for epoch in range(max_epoch):
                self.model.train()
                for (input_ids, attention_mask, decoder_input_ids, decoder_attention_mask) in tqdm(self.dataset['train'], total=len(self.dataset['train'])):
                    input_ids = input_ids.long().to(device)
                    attention_mask = attention_mask.long().to(device)
                    decoder_input_ids.contiguous()
                    decoder_attention_mask.contiguous()
                    labels = decoder_input_ids[..., 1:].long().to(device)
                    decoder_input_ids = decoder_input_ids[..., :-1].long().to(device)
                    decoder_attention_mask = decoder_attention_mask[..., :-1].long().to(device)
                    

                    loss = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, \
                        decoder_attention_mask=decoder_attention_mask, labels=labels)[0]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()
                print(f"epoch: {epoch+1}, loss: {loss.item()}")
                if eval==True:
                    self.model.eval()
                    total_len = len(self.dataset['valid']) * self.args.batch_size
                    rouge = load_metric('rouge')
                    rouge1 = 0
                    rouge2 = 0
                    rougeL = 0
                    for (input_ids, attention_mask, references) in self.dataset['valid']:
                        input_ids = input_ids.long().to(device)
                        attention_mask = attention_mask.long().to(device)
                        references = list(references[0]) # tuple to list
                        sum_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=62, early_stopping=True)
                        summaries = self.tokenizer.batch_decode(sum_ids, skip_special_tokens=True)
                        score = rouge._compute(predictions=summaries, references=references, use_agregator=False)
                        r1_score = np.array(score['rouge1'])
                        r2_score = np.array(score['rouge2'])
                        rL_score = np.array(score['rougeL'])
                        rouge1 += np.sum(r1_score, axis=0)[2]
                        rouge2 += np.sum(r2_score, axis=0)[2]
                        rougeL += np.sum(rL_score, axis=0)[2]
                    rouge1 /= total_len
                    rouge2 /= total_len
                    rougeL /= total_len
                    print(f"rouge1: {rouge1}, rouge2: {rouge2}, rougeL: {rougeL}")

    def save_model(self):
        model_save_path = f"{self.args.model_save_path}/max_epochs={self.args.max_epoch}_lr={self.args.lr}_batch_size={self.args.batch_size}"
        print(f"model_save_path: {model_save_path}")
        self.model.save_pretrained(model_save_path)
