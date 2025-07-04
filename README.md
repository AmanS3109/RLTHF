# 🧠 RLHF Reward Model

This project implements a reward model trained using Reinforcement Learning from Human Feedback (RLHF). It learns to rank LLM outputs based on human preferences using pairwise comparison data.

---

## 📌 Key Features

- ✅ Transformer-based reward model (BERT)
- ✅ Pairwise comparison loss (r_chosen > r_rejected)
- ✅ Evaluation on human preference alignment
- ✅ FastAPI for scoring new completions
- ✅ Reward distribution visualizations
- ✅ Fully reproducible via config + requirements

---

## 🚀 Setup Instructions

### 1. Clone the Repo & Setup Environment
```bash
git clone <your-private-or-public-link>
cd reward_model_rlhf

python -m venv rl_env
.\rl_env\Scripts\activate  # (Windows)
# or
source rl_env/bin/activate  # (Linux/Mac)

pip install -r requirements.txt
