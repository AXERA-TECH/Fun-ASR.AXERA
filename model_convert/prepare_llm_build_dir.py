import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = REPO_ROOT / "model_repo"
DEFAULT_QWEN_DIR = MODEL_REPO / "Qwen3-0.6B"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Qwen3-0.6B-LLM-Build"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a standalone HF-style Qwen3-0.6B directory for pulsar2 llm_build."
    )
    parser.add_argument("--model-repo", default=str(MODEL_REPO), help="Fun-ASR model repo.")
    parser.add_argument("--qwen-dir", default=str(DEFAULT_QWEN_DIR), help="Base Qwen3 tokenizer/config dir.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Prepared LLM build directory.")
    parser.add_argument("--checkpoint", default="", help="Optional explicit checkpoint path. Defaults to model_repo/model.pt.")
    return parser.parse_args()


def load_state_dict(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def collect_llm_weights(state_dict):
    llm_weights = {}
    prefix = "llm."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            llm_weights[key[len(prefix) :]] = value
    if not llm_weights:
        raise ValueError("No llm.* weights were found in the checkpoint.")
    return llm_weights


def copy_base_files(qwen_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    required = [
        "config.json",
        "generation_config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    for name in required:
        src = qwen_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing base Qwen file: {src}")
        shutil.copy2(src, output_dir / name)


def maybe_update_dtype(config_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["torch_dtype"] = "bfloat16"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def save_llm_weights(output_dir: Path, llm_weights):
    bin_path = output_dir / "pytorch_model.bin"
    torch.save(llm_weights, bin_path)
    safetensors_path = output_dir / "model.safetensors"
    save_file(llm_weights, str(safetensors_path))
    return bin_path, safetensors_path


def main():
    args = parse_args()
    model_repo = Path(args.model_repo).resolve()
    qwen_dir = Path(args.qwen_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else (model_repo / "model.pt")

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not qwen_dir.exists():
        raise FileNotFoundError(f"Base Qwen dir not found: {qwen_dir}")

    state_dict = load_state_dict(checkpoint)
    llm_weights = collect_llm_weights(state_dict)
    copy_base_files(qwen_dir, output_dir)
    maybe_update_dtype(output_dir / "config.json")
    weight_path, safetensors_path = save_llm_weights(output_dir, llm_weights)

    print(f"[OK] Prepared llm_build directory: {output_dir}")
    print(f"[OK] Saved LLM weights: {weight_path}")
    print(f"[OK] Saved LLM safetensors: {safetensors_path}")
    print(f"[INFO] Number of llm tensors: {len(llm_weights)}")


if __name__ == "__main__":
    main()
