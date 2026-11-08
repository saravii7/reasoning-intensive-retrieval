# Reasoning-Intensive Retrieval

A search system that finds the right reference passages for **reasoning-heavy biology questions**.
Built for the **Olo Academy Reasoning-Intensive Retrieval** Kaggle competition, part of the AI Engineering Bootcamp (AIEP), Olo Academy in partnership with MCIT.

**Result: ranked 3rd of 21 teams, leaderboard score 0.60.** Team of 2.

## The problem

- A user asks a biology question; the system must return the 10 most relevant passages from a corpus of about **49,000** passages.
- **Metric:** nDCG@10 (relevant passages near the top of the list count more).
- **The hard part:** the correct passage often shares almost no words with the question. A question about why insects fly toward light needs a passage about *phototaxis*, which the question never mentions. Plain keyword search misses it.

## Approach

| Step | What it does | Training nDCG@10 |
|---|---|---|
| 1. BM25 | Keyword search (implemented from scratch with scikit-learn) | ~11 |
| 2. LLM query expansion | **Qwen3-4B** rewrites each question as a short scientific explanation with the key terms, then we search with that text | **~45** |
| 3. Semantic search | **Qwen3-Embedding-0.6B** embeddings, so passages match by meaning, not only by words | added as a channel |
| 4. Fusion | Normalized scores from keyword and semantic search, with weights tuned on training questions | **0.60 on the leaderboard** |
| 5. Reranker (tested) | bge-reranker-v2-m3 on the top 100 passages | lowered the score, so it was **not used** |

### Key decisions
- **Explain the question before searching:** the biggest gain came from letting the LLM explain the question in scientific terms. The LLM is used to describe, not to answer.
- **Match training to test:** training questions are long and test questions are short, so training questions were shortened to the test length before choosing the fusion weights.
- **Keep only what helps:** the reranker was tested and dropped because it reduced the score on training questions.

## How to run

1. Open the notebook in Kaggle and attach the competition data.
2. Settings: **Accelerator: GPU T4 x2** and **Internet: On** (to download the models).
3. Run the cells in order. The notebook writes `submission.csv`.

```
pip install -r requirements.txt
```

## Files

- `reasoning_intensive_retrieval.ipynb`: full pipeline (comments in Arabic)
- `requirements.txt`: Python packages

Competition data is not included.

## Tech stack

Python, BM25, Qwen3-4B (LLM), Qwen3-Embedding, sentence-transformers, Hugging Face Transformers, PyTorch, scikit-learn, Kaggle Notebooks (GPU)

## Author

**Sarah Al-Bahesh** · [LinkedIn](https://www.linkedin.com/in/sarah-albahesh) · [GitHub](https://github.com/saravii7)
