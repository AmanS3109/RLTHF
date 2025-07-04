# api/score_api.py

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from model.model import RewardModel

app = FastAPI()

# -----------------------------
# Config
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-uncased"
checkpoint_path = "checkpoints/reward_model_epoch3.pt"
max_length = 256

# -----------------------------
# Load Model + Tokenizer
# -----------------------------
try:
    model = RewardModel(model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✅ Model and tokenizer loaded successfully.")
except Exception as e:
    print("❌ Failed to load model or tokenizer:", e)

# -----------------------------
# Request Schema
# -----------------------------
class ScoreRequest(BaseModel):
    completion: str

# -----------------------------
# Scoring Endpoint
# -----------------------------
@app.post("/score")
def score_text(req: ScoreRequest):
    try:
        # Tokenize the input
        inputs = tokenizer(
            req.completion,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
            add_special_tokens=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Print debug info
        print("🔍 Tokenized input keys:", inputs.keys())

        # Inference
        with torch.no_grad():
            reward = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            score = reward.item()

        print(f"📤 Score: {score:.4f}")
        return {"reward_score": round(score, 4)}

    except Exception as e:
        print("❌ Error during scoring:", str(e))
        return {"error": str(e)}
