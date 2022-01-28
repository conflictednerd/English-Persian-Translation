import os

import numpy as np
import pandas as pd
import torch
from datasets import load_metric
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AdamW, AutoConfig, AutoTokenizer,
                          DataCollatorWithPadding, DataCollatorForSeq2Seq, MBart50TokenizerFast,
                          MBartForConditionalGeneration, Trainer,
                          TrainingArguments)

from translator import Translator
import pickle


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, english_encodings, farsi_encodings):
        self.english_encodings = english_encodings
        self.farsi = farsi_encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.english_encodings.items()}
        item['labels'] = torch.tensor(self.farsi['input_ids'][idx])
        return item

    def __len__(self):
        return len(self.farsi['input_ids'])


class LMTranslator(Translator):
    def __init__(self, args):
        super().__init__()
        self.CLEANIFY = False
        self.train_data = []
        self.test_data = []
        self.DEVICE = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(
            self.DEVICE) if not args.load_model else self.load()
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt",
                                                              src_lang='en_XX', tgt_lang='fa_IR')
        self.load_train_data(args)

        def freeze_params(model):
            for par in model.parameters():
                par.requires_grad = False

        print("Before freezing, number of parameters: ",
              sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        freeze_params(self.model.get_encoder())  ## freeze the encoder
        # freeze_params(self.model.get_decoder())  ## freeze the decoder

        if args.train:
            self.train(args)

        if args.test:
            self.test(args)

    def load(self, path: str):

        return

    def save(self, path: str):
        pass

    def get_Dataset(self, data):
        data_eng = self.tokenizer(data['english'].tolist())
        data_farsi = self.tokenizer(data['farsi'].tolist())
        return TranslationDataset(data_eng, data_farsi)

    def save_object(self, obj, path):
        with open('data/' + path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_object(self, path):
        with open('data/' + path, 'rb') as f:
            return pickle.load(f)

    def load_train_data(self, args):
        path = args.data_path
        farsi = ''
        with open(os.path.join(path, 'mizan_fa.txt'), 'r', encoding='utf-8') as f:
            farsi += self.clean_fa(f.read()) if self.CLEANIFY else f.read()

        english = ''
        with open(os.path.join(path, 'mizan_en.txt'), 'r', encoding='utf-8') as f:
            english += f.read()

        farsi_parags = farsi.split('\n')
        english_parags = english.split('\n')

        all_data = {'farsi': farsi_parags, 'english': english_parags}
        df = pd.DataFrame(all_data)

        _, initial_df = train_test_split(df, test_size=0.025)
        train_data, test_data = train_test_split(initial_df, test_size=0.01)
        self.train_data = self.get_Dataset(train_data)
        self.test_data = self.get_Dataset(test_data)
        self.test_unpolished = test_data
        if args.over_write_test:
            self.save_object(test_data, "test_df.pkl")

    def translate(self, src_txt: str):
        pass

    def train(self, args):

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, self.model, padding=True)
        train_dataloader = DataLoader(self.train_data, batch_size=4, collate_fn=data_collator,
                                      shuffle=False)

        gradient_accumulations = 6

        self.model.train()  ## set the train mode
        print("After freezing, number of parameters: ",
              sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        print(len(train_dataloader))

        optimizer = AdamW(self.model.parameters(), lr=2e-05)

        for epoch in range(int(3)):
            self.model.train()
            print("##### epoch no: " + str(epoch + 1))

            for batch_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

                X1 = batch['input_ids'].to(self.DEVICE)
                y1 = batch['labels'].to(self.DEVICE)
                pred = self.model(X1, attention_mask=batch['attention_mask'].to(
                    self.DEVICE), labels=y1)

                loss = pred.loss / gradient_accumulations
                loss.backward()

                if (batch_idx + 1) % gradient_accumulations == 0:
                    optimizer.step()
                    self.model.zero_grad()

            self.model.save_pretrained(args.models_dir)

    def test(self, args):

        test_unpolished = self.load_object("test_df.pkl")

        metric = load_metric("sacrebleu")

        self.model.eval()
        with torch.no_grad():
            for index, row in tqdm(test_unpolished.iterrows()):
                encoded_eng = self.tokenizer(row['english'], return_tensors="pt").to(self.DEVICE)

                generated_tokens = self.model.generate(**encoded_eng,
                                                       forced_bos_token_id=self.tokenizer.lang_code_to_id["fa_IR"])

                model_predictions = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                ref = [[row['farsi']]]
                pred = [model_predictions[0]]
                metric.add_batch(predictions=pred, references=ref)

        final_score = metric.compute()

        print(final_score)
