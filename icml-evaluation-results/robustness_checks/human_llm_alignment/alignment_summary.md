# Human ↔ LLM Alignment (Single-turn)

Overall accuracy (decisive only): 0.538
Pearson r: 0.178
Spearman r: 0.210
AUC: 0.629

By baseline:
{
  "gpt5": {
    "n_total": 75,
    "n_decisive": 71,
    "accuracy": 0.4507042253521127,
    "pearson_r": 0.1502429104300974,
    "spearman_r": 0.1490860580877007,
    "auc": 0.5879629629629629
  },
  "claude": {
    "n_total": 79,
    "n_decisive": 78,
    "accuracy": 0.7435897435897436,
    "pearson_r": 0.18130384319427778,
    "spearman_r": 0.23227936553435188,
    "auc": 0.6693121693121693
  },
  "gemini": {
    "n_total": 64,
    "n_decisive": 61,
    "accuracy": 0.3770491803278688,
    "pearson_r": 0.008226420676494562,
    "spearman_r": 0.1110640140858815,
    "auc": 0.5626361655773421
  }
}
