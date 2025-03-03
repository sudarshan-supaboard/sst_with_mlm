import torch

from model import tokenizer, model

def predict(input_text: str):
  # tokenize the text
  input_ids = tokenizer.encode(input_text, return_tensors="pt")

  # get the logits
  with torch.no_grad():
    logits = model(input_ids).logits

  # extract masked_token_id
  masked_token_id = tokenizer.mask_token_id
  masked_token_logits = logits[0, input_ids[0] == masked_token_id, :]

  # get top 5 words
  top_5_tokens = torch.topk(masked_token_logits, 5, dim=1).indices[0].tolist()
  top_5_words = [tokenizer.decode([token]) for token in top_5_tokens]

  return top_5_words
