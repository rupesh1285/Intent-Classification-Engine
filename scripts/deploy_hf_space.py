"""
Create/update the Hugging Face Space from deploy/hf-space.

Needs a write token:
  setx HF_TOKEN "hf_..."   (new terminal after setx)
  OR:  .\\.venv\\Scripts\\huggingface-cli.exe login

Then:
  .\\.venv\\Scripts\\python.exe -m scripts.deploy_hf_space
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, get_token, whoami

ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = ROOT / "deploy" / "hf-space"
DEFAULT_SPACE = "Intent-Classification-Engine"


def main() -> None:
    token = get_token() or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("NEED_LOGIN")
        print("Create a write token: https://huggingface.co/settings/tokens")
        print("Then either:")
        print('  setx HF_TOKEN "hf_..."   # restart terminal')
        print("  OR run: .\\.venv\\Scripts\\huggingface-cli.exe login")
        print("After that re-run: python -m scripts.deploy_hf_space")
        sys.exit(2)

    user = whoami(token=token)["name"]
    space_id = os.environ.get("HF_SPACE_ID", f"{user}/{DEFAULT_SPACE}")
    print(f"deploying Space -> {space_id}")
    print(f"from folder    -> {SPACE_DIR}")

    if not (SPACE_DIR / "app.py").exists():
        raise SystemExit(f"missing {SPACE_DIR / 'app.py'}")
    if not (SPACE_DIR / "models" / "intent_classifier.joblib").exists():
        raise SystemExit("missing Space model: deploy/hf-space/models/intent_classifier.joblib")

    api = HfApi(token=token)
    create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True, token=token)
    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=space_id,
        repo_type="space",
        commit_message="Deploy Intent Classification Engine Gradio Space",
    )
    url = f"https://huggingface.co/spaces/{space_id}"
    print("DONE")
    print(url)
    (ROOT / "deploy" / "SPACE_URL.txt").write_text(url + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
