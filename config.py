import torch

class Config:
    PROJECT_ID="true-sprite-412217"
    BUCKET_NAME="pleasedontbankrupt"
    OUTPUT_DIR="bert_checkpoints"
    MODEL_PATH="FacebookAI/roberta-base"
    DATASET_PATH="sudarshan1927/go-emotions-and-generated"
    RANDOM_STATE=42
    PROJECT_NAME="sst_with_mlm"
    MASK_TOKEN="[MASK]"
   
    @classmethod
    def device(cls):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def set_roberta_model(cls):
        cls.MODEL_PATH  = "FacebookAI/roberta-base"
        cls.OUTPUT_DIR = "roberta_checkpoints"
        cls.MASK_TOKEN = "<mask>"
    
    @classmethod
    def set_bert_model(cls):
        cls.MODEL_PATH = "google-bert/bert-base-uncased"
        cls.OUTPUT_DIR = "bert_checkpoints"
        cls.MASK_TOKEN = "[MASK]"
    
    