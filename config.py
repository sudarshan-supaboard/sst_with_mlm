import torch

class Config:
    PROJECT_ID="true-sprite-412217"
    BUCKET_NAME="pleasedontbankrupt"
    OUTPUT_DIR="checkpoints"
    MODEL_PATH="google-bert/bert-base-uncased"
    DATASET_PATH="sudarshan1927/go-emotions-and-generated"
    RANDOM_STATE=42
    PROJECT_NAME="sst_with_mlm"
   
    @classmethod
    def device(cls):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
