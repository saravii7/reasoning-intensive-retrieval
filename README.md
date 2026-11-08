# Reasoning-Intensive Retrieval

> *"Why do moths fly toward light?"* The passage that answers it talks about **phototaxis**, a word the question never uses. Keyword search can't find it. This project teaches search to **think about the question first**.

🏆 **3rd of 21 teams** · Olo Academy Kaggle competition (AIEP bootcamp, Olo Academy × MCIT)
📈 **nDCG@10: ~11 → ~45** on training questions (BM25 → LLM-expanded search) · **0.60** on the leaderboard

## The challenge
- Find the 10 best passages for each biology question in a **49,000-passage** corpus
- Scored by **nDCG@10**: the right passage near the top counts most
- The answer often shares almost no words with the question

## How it works
```
question ──► Qwen3-4B writes a short scientific explanation
               │
               ├──► BM25 keyword search ─────┐
               └──► Qwen3-Embedding search ──┴──► weighted fusion ──► top 10
```

| Step | Result |
|---|---|
| BM25 on the raw question | ~11 nDCG@10 |
| + **LLM query expansion** (Qwen3-4B) | **~45** |
| + semantic search + tuned fusion | **0.60** leaderboard |
| Reranker (bge-reranker-v2-m3) | lowered the score → **dropped** |

**Design choices**
- **The LLM explains, it doesn't answer:** its job is to add the scientific terms the passages use.
- **Match train to test:** training questions were shortened to test length before tuning the fusion weights.
- **Keep only what helps:** the reranker looked promising but hurt the score, so it was cut.

## Run it
1. Open `reasoning_intensive_retrieval.ipynb` on Kaggle and attach the competition data.
2. Settings: **GPU T4 x2**, **Internet on** (to download the models). Run all cells.

**Stack:** Python · Qwen3-4B · Qwen3-Embedding · sentence-transformers · Hugging Face · BM25 · PyTorch

Competition data is not included. Team of 2.

**Sarah Al-Bahesh** · [LinkedIn](https://www.linkedin.com/in/sarah-albahesh) · [GitHub](https://github.com/saravii7)
