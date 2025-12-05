"""
Training script for Email Classification Model.

Note:
The actual training was performed in Google Colab due to local hardware limitations.
This script documents the training pipeline and can be executed on GPU-enabled systems.

"""

import torch
import pandas as pd

from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from parser import parse_raw_email_csv

parse_raw_email_csv('../data/cat_emails_v2(in).csv', '../data/clean_email.csv')

data = pd.read_csv('../data/clean_email.csv')

data['label'] = data['category'].astype('category').cat.codes

train_data, test_data = train_test_split(data, test_size = 500, random_state = 42, stratify = data['label'])
train_dataset = Dataset.from_pandas(train_data)
test_dataset  = Dataset.from_pandas(test_data)

model_name = 'distilbert-base-german-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch['email_text'], truncation = True, padding = 'max_length', max_length = 256)

train_dataset = train_dataset.map(tokenize, batched = True)
test_dataset = test_dataset.map(tokenize, batched = True)

train_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'label'])

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = data['label'].nunique())

device = torch.device('cuda')
model.to(device)

args = TrainingArguments(
    output_dir = 'models/base_email_model',
    learning_rate = 2e-5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    num_train_epochs = 10,
    weight_decay = 0.01,
    logging_steps = 50,
    save_strategy = 'epoch',
    eval_strategy = 'epoch',
    load_best_model_at_end = True,
    report_to = 'none'
)

trainer = Trainer(
    model = model,
    args = args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
)

trainer.train()
trainer.save_model('models/base_email_model')