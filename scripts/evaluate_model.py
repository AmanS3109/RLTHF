# scripts/evaluate_model.py

import torch
from torch.utils.data import DataLoader
from model.model import RewardModel
from data.prepare_dataset import PairwisePreferenceDataset
import os
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-uncased"
checkpoint_path = "checkpoints/reward_model_epoch3.pt"  # Update if needed
batch_size = 4
max_length = 256

# -----------------------------
# Load model
# -----------------------------
model = RewardModel(model_name)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Load dataset (use 'train' if no 'test')
# -----------------------------
val_dataset = PairwisePreferenceDataset(split="test", model_name=model_name, max_length=max_length)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# -----------------------------
# Evaluate
# -----------------------------
total = 0
correct = 0

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids_chosen = batch["input_ids_chosen"].to(device)
        attention_mask_chosen = batch["attention_mask_chosen"].to(device)
        input_ids_rejected = batch["input_ids_rejected"].to(device)
        attention_mask_rejected = batch["attention_mask_rejected"].to(device)

        reward_chosen = model(input_ids_chosen, attention_mask_chosen)
        reward_rejected = model(input_ids_rejected, attention_mask_rejected)

        correct += (reward_chosen > reward_rejected).sum().item()
        total += reward_chosen.size(0)

accuracy = correct / total * 100
print(f"✅ Evaluation complete. Accuracy: {accuracy:.2f}%")
