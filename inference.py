from typing import List
import torch
import pandas as pd
from config import Config
from transformers import BertForMaskedLM, BertTokenizer
from peft import peft_model
from pprint import pprint

from sklearn.metrics import accuracy_score, f1_score

checkpoint_uri = "./checkpoints/checkpoint-6000"
model = BertForMaskedLM.from_pretrained(Config.MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(Config.MODEL_PATH,
                                          do_lower_case=True,
                                          strip_accents=True)

model = peft_model.PeftModelForSequenceClassification.from_pretrained(model, checkpoint_uri)
model.eval()

def predict(inputs: List[str]):
  
  # tokenize the text
  input_ids = tokenizer(inputs, return_tensors="pt", padding="max_length")

  # get the logits
  with torch.no_grad():
    logits = model(input_ids=input_ids['input_ids'], attention_mask=input_ids["attention_mask"]).logits

  # extract masked_token_id
  masked_token_id = tokenizer.mask_token_id

  masked_token_logits = logits[:, masked_token_id, :]

  # get top 5 words
  top_tokens = torch.topk(masked_token_logits, 5, dim=1).indices
  

  
  top_words_1 = [tokenizer.decode([token]) for token in top_tokens[:,0]]
  # top_words_2 = [tokenizer.decode([token]) for token in top_tokens[:,1]]
  # top_words_3 = [tokenizer.decode([token]) for token in top_tokens[:,2]]
  # top_words_4 = [tokenizer.decode([token]) for token in top_tokens[:,3]]
  # top_words_5 = [tokenizer.decode([token]) for token in top_tokens[:,4]]
  

  return top_words_1


if __name__ == '__main__':
  df = pd.read_csv("./toy_store_test.csv")
  # df = df.iloc[:100]
  preds_1 = []
  # preds_2 = []
  # preds_3 = []
  # preds_4 = []
  # preds_5 = []
  
  labels = []
  
  for i in range(0, len(df['review']), 8):
    
    print(f'predicting from {i}-{i+8}')
    inputs = []
    for j in df['review'].iloc[i:i+8]:
      inputs.append(f"'{j}', emotion of the given text is [MASK]?")

    preds_1_ = predict(inputs)
    
    preds_1.extend(preds_1_)
    # preds_2.extend(preds_2_)
    # preds_3.extend(preds_3_)
    # preds_4.extend(preds_4_)
    # preds_5.extend(preds_5_)
    labels.extend(df['rating'].iloc[i:i+8].to_list())
    
    
    if i % 64 == 0:
      # print("1")
      # print(accuracy_score(y_true=labels, y_pred=preds_1))
      # print(f1_score(y_true=labels, y_pred=preds_1, average='weighted'))
  
      # print("\n2")
      # print(accuracy_score(y_true=labels, y_pred=preds_2))
      # print(f1_score(y_true=labels, y_pred=preds_2, average='weighted'))

      # print("\n3")
      # print(accuracy_score(y_true=labels, y_pred=preds_3))
      # print(f1_score(y_true=labels, y_pred=preds_3, average='weighted'))

      # print("\n4")
      # print(accuracy_score(y_true=labels, y_pred=preds_4))
      # print(f1_score(y_true=labels, y_pred=preds_4, average='weighted'))

      # print("\n5")
      # print(accuracy_score(y_true=labels, y_pred=preds_5))
      # print(f1_score(y_true=labels, y_pred=preds_5, average='weighted'))
      
      (pd.crosstab(labels, preds_1, normalize='index') * 100).to_csv("./confmat_toy.csv")
  
    
    
    
  # print("1")
  # print(accuracy_score(y_true=labels, y_pred=preds_1))
  # print(f1_score(y_true=labels, y_pred=preds_1, average='weighted'))
  
  # print("\n2")
  # print(accuracy_score(y_true=labels, y_pred=preds_2))
  # print(f1_score(y_true=labels, y_pred=preds_2, average='weighted'))
  
  # print("\n3")
  # print(accuracy_score(y_true=labels, y_pred=preds_3))
  # print(f1_score(y_true=labels, y_pred=preds_3, average='weighted'))
  
  # print("\n4")
  # print(accuracy_score(y_true=labels, y_pred=preds_4))
  # print(f1_score(y_true=labels, y_pred=preds_4, average='weighted'))
  
  # print("\n5")
  # print(accuracy_score(y_true=labels, y_pred=preds_5))
  # print(f1_score(y_true=labels, y_pred=preds_5, average='weighted'))
  (pd.crosstab(labels, preds_1, normalize='index') * 100).to_csv("./confmat_toy.csv")
  