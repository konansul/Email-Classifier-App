import os
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

data = pd.read_csv('../data/clean_email.csv')
data['label'] = data['category'].astype('category').cat.codes

categories: list[str] = list(map(str, data['category'].astype('category').cat.categories))
label_to_category: dict[int, str] = dict(enumerate(categories))

train_data, test_data = train_test_split(data, test_size = 500, random_state = 42, stratify = data['label'] )

test_dataset = Dataset.from_pandas(test_data)

model_path = '../models/base_email_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)

def tokenize(batch):
    return tokenizer(batch['email_text'], truncation = True, padding = 'max_length', max_length = 256)

test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'label'])

# Creating trainer only for prediction
args = TrainingArguments(output_dir = 'eval_tmp', report_to = 'none')

trainer = Trainer(model = model, args = args)

predictions_logits = trainer.predict(test_dataset)

predictions = np.argmax(predictions_logits.predictions, axis = 1)
labels = predictions_logits.label_ids


# Classification Report
report = classification_report(labels, predictions, target_names = list(label_to_category.values()))
print(report)

os.makedirs('../reports', exist_ok=True)

with open('../reports/classification_report.txt', 'w', encoding = 'utf-8') as f:
    f.write('Classification Report\n')
    f.write(report)

print('\nClassification report saved to: ../reports/classification_report.txt\n')


# Sample of errors (top 20)
test_texts = test_data['email_text'].tolist()
test_true_labels = test_data['label'].tolist()

errors = [ ]

for text, true, pred in zip(test_texts, test_true_labels, predictions):
    if true != pred:
        errors.append({'true': label_to_category[true], 'pred': label_to_category[pred] })

errors_data = pd.DataFrame(errors)
print('\n Sample of errors (top 20) \n')
print(errors_data.head(20))

with open('../reports/misclassified_sample_top_20.txt', 'w', encoding = 'utf-8') as f:
    f.write('Misclassified Sample (Top 20)\n')

    for i, row in errors_data.head(20).iterrows():
        f.write(f"True: {row['true']} | Predicted: {row['pred']}\n")

print('\nMisclassified samples saved to: ../reports/misclassified_samples_top_20.txt\n')


# Confusion Matrix
plt.figure(figsize = (18, 14))
sns.heatmap(confusion_matrix(labels, predictions), annot = True, cmap = 'Blues',
            xticklabels = list(label_to_category.values()), yticklabels = list(label_to_category.values()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation = 90)
plt.yticks(rotation = 0)
plt.tight_layout()
plt.savefig('../reports/confusion_matrix.png')
plt.show()