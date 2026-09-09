---
title: Intent Classification Engine
emoji: 🔒
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: Benign vs prompt-injection classifier demo
---

# Intent Classification Engine

Live Gradio demo for **benign** vs **injection** (prompt-injection / jailbreak) classification.

- GitHub: [rupesh1285/Intent-Classification-Engine](https://github.com/rupesh1285/Intent-Classification-Engine)
- Space ships the calibrated **TF-IDF + LinearSVC** model (fast on free CPU).
- Full local stack (MiniLM + DistilBERT + OOF stacking, ~91.6% acc) runs in the repo.
