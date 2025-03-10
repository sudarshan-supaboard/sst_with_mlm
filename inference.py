import torch
import pandas as pd

from typing import List
from config import Config
from transformers import BertForMaskedLM, BertTokenizer
from peft import peft_model
from sklearn.metrics import accuracy_score
from pprint import pprint
checkpoint_uri = "./checkpoints/checkpoint-6000"
model = BertForMaskedLM.from_pretrained(Config.MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(Config.MODEL_PATH,
                                          do_lower_case=True,
                                          strip_accents=True)

model = peft_model.PeftModelForSequenceClassification.from_pretrained(model, checkpoint_uri)
model.eval()

def predict(inputs: List[str]):
  
  # tokenize the text
  input_ids = tokenizer(inputs, return_tensors="pt", padding="max_length", truncation=True)

  # get the logits
  with torch.no_grad():
    logits = model(input_ids=input_ids['input_ids'], attention_mask=input_ids["attention_mask"]).logits

  # extract masked_token_id
  masked_token_id = tokenizer.mask_token_id

  masked_token_logits = logits[:, masked_token_id, :]

  # get top 5 words
  top_tokens = torch.topk(masked_token_logits, 5, dim=1).indices

  top_words = [tokenizer.decode([token]) for token in top_tokens[:,0]]

  return top_words


if __name__ == '__main__':
  df = pd.read_csv("./toy_store_test.csv")
  # df = df.iloc[:100]

  preds = []
  labels = []
  
  for i in range(0, len(df['review']), 8):
    
    print(f'predicting from {i}-{i+8}')
    inputs = []
    for j in df['review'].iloc[i:i+8]:
      inputs.append(f"The emotion in the text '{j}' is [MASK]?")

    preds.extend(predict(inputs))
    labels.extend(df['rating'].iloc[i:i+8].to_list())
    
    if i % 64 == 0:
      print(accuracy_score(y_true=labels, y_pred=preds))
      (pd.crosstab(labels, preds, normalize='index') * 100).to_csv("./confmat_toy.csv")
  
  
  print(accuracy_score(y_true=labels, y_pred=preds))
  (pd.crosstab(labels, preds, normalize='index') * 100).to_csv("./confmat_toy.csv")
  