import torch
import gc
import shutil
import os
import GPUtil
import torch.distributed as dist

from transformers import TrainerCallback
from google.cloud import storage
from config import Config

def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0 

def clear_cache():
  torch.cuda.empty_cache()  # Free unused memory
  gc.collect()  # Run Python garbage collector

def get_memory_usage():
    # Get all GPUs
    gpus = GPUtil.getGPUs()
    # Print details for each GPU
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.name}")
        print(f"  Total Memory: {gpu.memoryTotal} MB")
        print(f"  Used Memory: {gpu.memoryUsed} MB")
        print(f"  Free Memory: {gpu.memoryFree} MB")
        print(f"  GPU Utilization: {gpu.load * 100:.1f}%")
        print("-" * 40)


def upload_checkpoints():
    storage_client = storage.Client(project=Config.PROJECT_ID)
    bucket = storage_client.bucket(Config.BUCKET_NAME)
    
    zip_file = f"{Config.OUTPUT_DIR}.zip"
    shutil.make_archive(Config.OUTPUT_DIR, 'zip', Config.OUTPUT_DIR)
    print(f"Zipped {Config.OUTPUT_DIR} -> {zip_file}")
        
    blob = bucket.blob(zip_file)
    blob.upload_from_filename(zip_file)
    print(f"Uploaded {Config.OUTPUT_DIR} to gs://{Config.OUTPUT_DIR}/{zip_file}")


def get_tensor_memory(x):
    return (x.nelement() * x.element_size()) / (1024 * 1024 * 1024)

class Accuracy:
    def __init__(self):
        self.total_matches = 0
        self.total_counts = 0

    def add(self,preds, labels):
        self.total_matches += torch.eq(preds, labels).sum().item()
        self.total_counts += len(preds)

    def compute(self):
        return self.total_matches / self.total_counts
        
    def reset(self):
        self.total_matches = 0  # Reset the total matches
        self.total_counts = 0  # Reset the total counts

class AverageAccuracy:
    def __init__(self) -> None:
        self.total_matches = 0
        self.total_counts = 0
    
    def add(self, logits, labels):
        preds = torch.topk(logits, 5, dim=-1).indices
        self.total_matches += torch.eq(preds[:, 0], labels).sum().item()
        self.total_matches += torch.eq(preds[:, 1], labels).sum().item() * 0.8
        self.total_matches += torch.eq(preds[:, 2], labels).sum().item() * 0.6
        self.total_matches += torch.eq(preds[:, 3], labels).sum().item() * 0.4
        self.total_matches += torch.eq(preds[:, 4], labels).sum().item() * 0.2
        self.total_counts += len(labels)
        
    def compute(self):
        return self.total_matches / self.total_counts
    
    def reset(self):
        self.total_matches = 0
        self.total_counts = 0
    
class EarlyStoppingTrainingLossCallback(TrainerCallback):
    def __init__(self, patience=3, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def on_save(self, args, state, control, **kwargs):
        """Called at the end of every step to monitor training loss."""
        print(f"Running Early Stop Callback")
        if state.log_history:
            val_losses = [log["eval_loss"] for log in state.log_history if "eval_loss" in log]

            if val_losses:
                current_loss = val_losses[-1]  # Get the most recent training loss
                if current_loss < self.best_loss - self.min_delta:
                    self.best_loss = current_loss
                    self.counter = 0  # Reset patience counter if loss improves
                else:
                    self.counter += 1  # Increment counter if no improvement

                if self.counter >= self.patience:
                    control.should_training_stop = True
                    print("\nEarly stopping triggered due to no improvement in validation loss!")



class GCSUploadCallback(TrainerCallback):
    def __init__(self, bucket_name=Config.BUCKET_NAME, checkpoint_dir=Config.OUTPUT_DIR, bkt_upload=True):
        self.bucket_name = bucket_name
        self.checkpoint_dir = checkpoint_dir  # Local directory where checkpoints are saved
        self.storage_client = storage.Client(project=Config.PROJECT_ID)
        self.bucket = self.storage_client.bucket(bucket_name)
        self.bkt_upload = bkt_upload

    def upload_to_gcs(self, local_path):
        """Uploads a file or directory to GCS recursively."""
        zip_file = f"{local_path}.zip"
        shutil.make_archive(local_path, 'zip', local_path)
        print(f"Zipped {local_path} -> {zip_file}")
        
        blob = self.bucket.blob(zip_file)
        blob.upload_from_filename(zip_file)
        print(f"Uploaded {local_path} to gs://{self.bucket_name}/{zip_file}")
        
        return zip_file
        
    def on_save(self, args, state, control, **kwargs):
        """Triggered whenever a checkpoint is saved."""
        
        if not self.bkt_upload:
            print("Not uploading to GCS")
            return
        
        print("Saving checkpoint, uploading to GCS...")
        
        # Upload the entire checkpoint directory
        uploaded_file = self.upload_to_gcs(self.checkpoint_dir)

        print(f"Checkpoint uploaded to gs://{self.bucket_name}/{uploaded_file}")


class FreeMemoryCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        clear_cache()

if __name__ == '__main__':
    acc = AverageAccuracy()
    acc1 = Accuracy()
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])
    labels = torch.tensor([4, 2])
    acc.add(logits, labels)
    
    acc1.add(torch.argmax(logits, dim=-1), labels)
    
    print(acc.compute())
    acc.reset()
    
    print(acc1.compute())
    acc1.reset()