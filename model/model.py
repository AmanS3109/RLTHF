# model/model.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class RewardModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.reward_head = nn.Linear(self.transformer.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # shape: [batch_size, seq_len, hidden_size]
        cls_embedding = last_hidden_state[:, 0]        # use [CLS] token
        reward = self.reward_head(cls_embedding)       # shape: [batch_size, 1]
        return reward.squeeze(-1)                      # shape: [batch_size]
