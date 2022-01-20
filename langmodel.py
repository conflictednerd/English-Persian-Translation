from translator import Translator
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_metric
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoTokenizer,
                          DataCollatorWithPadding, Trainer,
                          TrainingArguments, AdamW)
import numpy as np


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, english_encodings, farsi_encodings):
        self.english_encodings = english_encodings
        self.farsi = farsi_encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.english_encodings.items()}
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
            self.DEVICE)
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt",
                                                              src_lang='en_XX', tgt_lang='fa_IR')
        if args.train:
            self.load_train_data(args.data_path)

        if args.test:
            self.test(args)

    def load(self, path: str):
        pass

    def save(self, path: str):
        pass

    def get_Dataset(self, data):
        data_eng = self.tokenizer(data['english'].tolist(), padding=True)
        data_farsi = self.tokenizer(data['farsi'].tolist(), padding=True)
        return TranslationDataset(data_eng, data_farsi)

    def load_train_data(self, path):
        farsi = ''
        with open(os.path.join(path, 'mizan_fa.txt'), 'r', encoding='utf-8') as f:
            farsi += self.clean(f.read()) if self.CLEANIFY else f.read()

        english = ''
        with open(os.path.join(path, 'mizan_en.txt'), 'r', encoding='utf-8') as f:
            english += f.read()

        farsi_parags = farsi.split('\n')
        english_parags = english.split('\n')

        all_data = {'farsi': farsi_parags, 'english': english_parags}
        df = pd.DataFrame(all_data)

        _, initial_df = train_test_split(df, test_size=0.1)
        train_data, test_data = train_test_split(initial_df, test_size=0.1)
        self.train_data = self.get_Dataset(train_data)
        self.test_data = self.get_Dataset(test_data)

    def translate(self, src_txt: str):
        pass

    def train(self, args):
        pass

    def test(self, args):
        data_collator = DataCollatorWithPadding(self.tokenizer)
        test_dataloader = DataLoader(self.test_data, batch_size=4, collate_fn=data_collator,
                                     shuffle=False)
        # print(self.test_data[0])
        metric = load_metric("bleu")


        self.model.eval()

        print(len(test_dataloader))

        for batch_idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

            if batch_idx == 5:
                exit(0)
            X1 = batch['input_ids'].to(self.DEVICE)
            # # print(X1.shape)
            y1 = batch['labels'].to(self.DEVICE)
            pred = self.model(X1, attention_mask=batch['attention_mask'].to(self.DEVICE))  #TODO: how to manage Seq2Seq batches here?


        # for index, row in tqdm(self.test_data.iterrows(), total=len(self.test_data)):
        #     encoded_eng = self.tokenizer(row['english'], return_tensors="pt").to(self.DEVICE)
        #
        #     generated_tokens = self.model.generate(**encoded_eng,
        #                                            forced_bos_token_id=self.tokenizer.lang_code_to_id["fa_IR"])
        #
        #     model_predictions = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        #     ref = [[row['farsi'].split()]]
        #     pred = [model_predictions[0].split()]
        #     metric.add_batch(predictions=pred, references=ref)
        #
        # final_score = metric.compute()
        #
        # print(final_score)
