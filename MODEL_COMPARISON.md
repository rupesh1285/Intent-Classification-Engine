# Model comparison

Same clean split (`train.csv` / `test.csv`, seed 42).

| approach | accuracy | macro F1 | notes |
|----------|----------|----------|-------|
| all-MiniLM-L6-v2 + LogisticRegression | 0.8389 | 0.8359 | semantic baseline |
| TF-IDF word(1-3)+char(3-6) + LinearSVC | 0.8885 | 0.8850 | strong lexical baseline |
| OOF stacking TF-IDF+MiniLM → meta LogReg | 0.8897 | 0.8870 | stacked generalization |
| soft-vote 0.7 TF-IDF + 0.3 MiniLM | 0.8933 | 0.8901 | best classical combo |
| DistilBERT fine-tuned (1 epoch, CPU) | 0.9111 | 0.9079 | strong transformer alone |
| **OOF stacking TF-IDF+MiniLM+DistilBERT** | **0.9159** | **0.9134** | **current default** |

## What I learned

- Lexical signal is strong; DistilBERT alone jumped ~2 pts over soft-vote.
- 3-base stacking edged DistilBERT further (~0.92 acc) — classic + transformer together.
- For security productization, also tune `P(injection)` threshold — see `reports/threshold_analysis.md`.
- More DistilBERT epochs → use Colab GPU (`notebooks/COLAB_DISTILBERT.md`).

## Architecture story (resume / interview)

1. Clean noisy multi-source data  
2. Compare lexical vs embedding baselines  
3. Soft-vote + OOF stacking  
4. Fine-tune DistilBERT, then stack all three (best)  
5. Error analysis + threshold tradeoffs + Gradio demo  
