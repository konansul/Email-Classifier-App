import json
import torch

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = 'models/base_email_model'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

with open(f'{model_path}/label_to_category.json', 'r') as f:
    label_to_category = json.load(f)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)
model.eval()

app = FastAPI(title = 'Email Classification API')

class EmailRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_email(req: EmailRequest):
    encoded = tokenizer(
        req.text,
        truncation = True,
        padding = 'max_length',
        max_length = 512,
        return_tensors = 'pt'
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=-1).item()

    pred_label = label_to_category[str(pred_id)]

    return {
        'prediction_id': pred_id,
        'category': pred_label
    }