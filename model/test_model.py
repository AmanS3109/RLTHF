# model/test_model.py

from model import RewardModel
from transformers import AutoTokenizer
import torch

# Load model and tokenizer
model_name = "bert-base-uncased"
model = RewardModel(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Test input
text = "Explain how photosynthesis works."
inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

# Forward pass
with torch.no_grad():
    reward_score = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

# Output
print("Reward Score:", reward_score.item())
