# Deploy guide (updated)

## Hugging Face Spaces — blocked on free tier (Mar 2026+)

Creating a **Gradio** Space on free `cpu-basic` now returns **402 Payment Required** and asks for [HF PRO](https://huggingface.co/pro).

Options:
1. Pay HF PRO, then run `python -m scripts.deploy_hf_space`
2. Use **Render free** instead (recommended below)

## Security

If you pasted an HF token in chat, **revoke it now**:  
https://huggingface.co/settings/tokens → delete that token → create a new one only if needed.

## Render free (recommended)

Package already lives in `deploy/hf-space/` (Gradio + TF-IDF). Root `render.yaml` is ready.

### Your manual steps

1. Open https://render.com and sign up / login (GitHub login is easiest)
2. **New → Blueprint** (or Web Service)
3. Connect repo: `rupesh1285/Intent-Classification-Engine`
4. Render should read `render.yaml`
5. Deploy → copy the `*.onrender.com` URL

If Blueprint does not pick rootDir correctly, create a **Web Service** manually:

- Root directory: `deploy/hf-space`
- Build: `pip install -r requirements.txt`
- Start: `python app.py`
- Instance: Free

### After you connect GitHub on Render

Tell me **"render connected"** if anything fails — I will adjust config.

## What is already done locally

- `deploy/hf-space/` app + model + requirements
- `scripts/deploy_hf_space.py` (for HF PRO later)
- `render.yaml` for Render Blueprint
