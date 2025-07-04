# model/train.py

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
from torch.optim import AdamW
from model.model import RewardModel
from data.prepare_dataset import PairwisePreferenceDataset
import os
import random
import numpy as np
from tqdm import tqdm

# ------------------------------
# 1. Reproducibility
# ------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ------------------------------
# 2. Config
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model_name = "bert-base-uncased"
batch_size = 2
lr = 2e-5
epochs = 3
start_epoch = 2  # 👈 RESUME from Epoch 2
max_length = 256

# ------------------------------
# 3. Load Model & Tokenizer
# ------------------------------
model = RewardModel(model_name=model_name).to(device)

# 👇 Load checkpoint from Epoch 1
checkpoint_path = f"checkpoints/reward_model_epoch{start_epoch - 1}.pt"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"✅ Loaded checkpoint: {checkpoint_path}")
else:
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    exit()

# ------------------------------
# 4. Load Dataset
# ------------------------------
train_dataset = PairwisePreferenceDataset(split="train", model_name=model_name, max_length=max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ------------------------------
# 5. Optimizer & Scheduler
# ------------------------------
optimizer = AdamW(model.parameters(), lr=lr)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * (epochs - start_epoch + 1)
)

# ------------------------------
# 6. Loss + Training Loop
# ------------------------------
loss_fn = nn.BCEWithLogitsLoss()

def compute_pairwise_loss(reward_chosen, reward_rejected):
    logits = reward_chosen - reward_rejected
    labels = torch.ones_like(logits)
    return loss_fn(logits, labels)

model.train()
for epoch in range(start_epoch, epochs + 1):
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

    for batch in pbar:
        input_ids_chosen = batch["input_ids_chosen"].to(device)
        attention_mask_chosen = batch["attention_mask_chosen"].to(device)
        input_ids_rejected = batch["input_ids_rejected"].to(device)
        attention_mask_rejected = batch["attention_mask_rejected"].to(device)

        optimizer.zero_grad()
        reward_chosen = model(input_ids_chosen, attention_mask_chosen)
        reward_rejected = model(input_ids_rejected, attention_mask_rejected)
        loss = compute_pairwise_loss(reward_chosen, reward_rejected)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"✅ Epoch {epoch} complete. Avg loss: {avg_loss:.4f}")

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/reward_model_epoch{epoch}.pt")

print("✅ Training complete!")
