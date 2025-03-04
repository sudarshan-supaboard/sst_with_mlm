import torch

class Config:
    PROJECT_ID="true-sprite-412217"
    BUCKET_NAME="pleasedontbankrupt"
    OUTPUT_DIR="checkpoints"
    MODEL_PATH="google-bert/bert-base-uncased"
   
    @classmethod
    def device(cls):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
