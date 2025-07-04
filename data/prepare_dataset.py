# data/prepare_dataset.py

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class PairwisePreferenceDataset(Dataset):
    def __init__(self, split="train", model_name="bert-base-uncased", max_length=512):
        # Load the Anthropic HH-RLHF dataset
        self.dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        # Tokenize both chosen and rejected
        chosen_tokens = self.tokenizer(
            chosen,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        rejected_tokens = self.tokenizer(
            rejected,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids_chosen": chosen_tokens["input_ids"].squeeze(0),
            "attention_mask_chosen": chosen_tokens["attention_mask"].squeeze(0),
            "input_ids_rejected": rejected_tokens["input_ids"].squeeze(0),
            "attention_mask_rejected": rejected_tokens["attention_mask"].squeeze(0),
        }


def get_dataloader(split="train", batch_size=8, shuffle=True):
    dataset = PairwisePreferenceDataset(split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # Quick test
    loader = get_dataloader(batch_size=2)
    batch = next(iter(loader))
    for key, value in batch.items():
        print(f"{key}: shape {value.shape}")
