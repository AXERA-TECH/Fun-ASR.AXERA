import argparse
import tarfile
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DUMP_DIR = REPO_ROOT / "model_convert" / "dump"
CALIB_DIR = REPO_ROOT / "model_convert" / "calib_data"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Pulsar2 Numpy calibration datasets from dumped Fun-ASR tensors."
    )
    parser.add_argument("--dump-dir", default=str(DUMP_DIR), help="Directory containing dumped .pt tensors.")
    parser.add_argument("--calib-dir", default=str(CALIB_DIR), help="Output calibration directory.")
    return parser.parse_args()


def save_npy(value: torch.Tensor, path: Path):
    np.save(path, value.detach().cpu().numpy())


def add_to_tar(tar_path: Path, npy_path: Path):
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(npy_path, arcname=npy_path.name)


def make_dataset(name: str, tensor_file: Path, calib_dir: Path):
    value = torch.load(tensor_file, map_location="cpu")
    npy_path = calib_dir / f"{name}.npy"
    tar_path = calib_dir / f"{name}.tar.gz"
    save_npy(value, npy_path)
    add_to_tar(tar_path, npy_path)
    print(f"[OK] {tar_path}")


def main():
    args = parse_args()
    dump_dir = Path(args.dump_dir).resolve()
    calib_dir = Path(args.calib_dir).resolve()
    calib_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "speech": dump_dir / "speech.pt",
        "speech_lengths": dump_dir / "speech_lengths.pt",
        "audio_encoder_out": dump_dir / "audio_encoder_out.pt",
        "audio_encoder_out_lens": dump_dir / "audio_encoder_out_lens.pt",
    }
    for key, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Required dump tensor not found for {key}: {path}")

    for key, path in required.items():
        make_dataset(key, path, calib_dir)


if __name__ == "__main__":
    main()
