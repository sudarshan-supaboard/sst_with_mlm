import torch
import gc
import shutil
import os

from transformers import TrainerCallback
from google.cloud import storage
from config import Config

def clear_cache():
  torch.cuda.empty_cache()  # Free unused memory
  gc.collect()  # Run Python garbage collector

def get_memory_usage(gpu_id = 0):

    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory  # Total GPU memory
    allocated_memory = torch.cuda.memory_allocated(gpu_id)  # Memory currently allocated by PyTorch
    cached_memory = torch.cuda.memory_reserved(gpu_id)  # Memory reserved by the caching allocator
    
    print(f"Total GPU Memory [{gpu_id}]: {total_memory / 1024**3:.2f} GB")
    print(f"Allocated Memory [{gpu_id}]: {allocated_memory / 1024**3:.2f} GB")
    print(f"Cached Memory [{gpu_id}]: {cached_memory / 1024**3:.2f} GB")


def upload_checkpoints():
    storage_client = storage.Client(project=Config.PROJECT_ID)
    bucket = storage_client.bucket(Config.BUCKET_NAME)
    
    zip_file = f"{Config.OUTPUT_DIR}.zip"
    shutil.make_archive(Config.OUTPUT_DIR, 'zip', Config.OUTPUT_DIR)
    print(f"Zipped {Config.OUTPUT_DIR} -> {zip_file}")
        
    blob = bucket.blob(zip_file)
    blob.upload_from_filename(zip_file)
    print(f"Uploaded {Config.OUTPUT_DIR} to gs://{Config.OUTPUT_DIR}/{zip_file}")


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
            print(f'Validation losses: {val_losses}')
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
    def __init__(self, bucket_name=Config.BUCKET_NAME, checkpoint_dir=Config.OUTPUT_DIR):
        self.bucket_name = bucket_name
        self.checkpoint_dir = checkpoint_dir  # Local directory where checkpoints are saved
        self.storage_client = storage.Client(project=Config.PROJECT_ID)
        self.bucket = self.storage_client.bucket(bucket_name)

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
        print("Saving checkpoint, uploading to GCS...")
        
        # Upload the entire checkpoint directory
        uploaded_file = self.upload_to_gcs(self.checkpoint_dir)

        print(f"Checkpoint uploaded to gs://{self.bucket_name}/{uploaded_file}")


