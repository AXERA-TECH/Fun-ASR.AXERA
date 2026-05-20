import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from export_utils import DUMP_DIR, MODEL_REPO, load_model, resolve_example_input


REPO_ROOT = Path(__file__).resolve().parents[1]
ENCODER_ONNX = REPO_ROOT / "model_convert" / "fun_asr_audio_encoder.onnx"
ADAPTOR_ONNX = REPO_ROOT / "model_convert" / "fun_asr_audio_adaptor.onnx"


def parse_args():
    parser = argparse.ArgumentParser(description="Verify chained audio_encoder/audio_adaptor ONNX inference.")
    parser.add_argument("--model-dir", default=str(MODEL_REPO), help="Local Fun-ASR model snapshot directory.")
    parser.add_argument("--wav-path", default="", help="Input wav path used to build the example speech tensor.")
    parser.add_argument("--speech-npy", default="", help="Optional pre-extracted speech feature tensor (.npy).")
    parser.add_argument(
        "--speech-lengths-npy",
        default="",
        help="Optional speech lengths tensor (.npy), required with --speech-npy.",
    )
    parser.add_argument("--encoder-onnx", default=str(ENCODER_ONNX), help="audio_encoder ONNX path.")
    parser.add_argument("--adaptor-onnx", default=str(ADAPTOR_ONNX), help="audio_adaptor ONNX path.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="PyTorch validation device.")
    parser.add_argument("--dump-dir", default=str(DUMP_DIR), help="Optional location to save numpy dumps.")
    return parser.parse_args()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b) / denom)


def main():
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    encoder_onnx = Path(args.encoder_onnx).resolve()
    adaptor_onnx = Path(args.adaptor_onnx).resolve()
    dump_dir = Path(args.dump_dir).resolve()

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not encoder_onnx.exists():
        raise FileNotFoundError(f"Encoder ONNX not found: {encoder_onnx}")
    if not adaptor_onnx.exists():
        raise FileNotFoundError(f"Adaptor ONNX not found: {adaptor_onnx}")

    model, kwargs = load_model(model_dir, args.device)
    speech, speech_lengths = resolve_example_input(
        model,
        kwargs,
        speech_npy=args.speech_npy,
        speech_lengths_npy=args.speech_lengths_npy,
        wav_path=args.wav_path,
        device=args.device,
    )

    with torch.no_grad():
        pt_encoder_out, pt_encoder_out_lens = model.audio_encoder(speech, speech_lengths)
        pt_adaptor_out, pt_adaptor_out_lens = model.audio_adaptor(pt_encoder_out, pt_encoder_out_lens)

    encoder_sess = ort.InferenceSession(str(encoder_onnx), providers=["CPUExecutionProvider"])
    adaptor_sess = ort.InferenceSession(str(adaptor_onnx), providers=["CPUExecutionProvider"])

    speech_np = speech.detach().cpu().numpy()
    speech_lengths_np = speech_lengths.detach().cpu().numpy()
    ort_encoder_out, ort_encoder_out_lens = encoder_sess.run(
        None,
        {"speech": speech_np, "speech_lengths": speech_lengths_np},
    )
    ort_adaptor_out, ort_adaptor_out_lens = adaptor_sess.run(
        None,
        {
            "audio_encoder_out": ort_encoder_out,
            "audio_encoder_out_lens": ort_encoder_out_lens,
        },
    )

    dump_dir.mkdir(parents=True, exist_ok=True)
    np.save(dump_dir / "onnx_audio_encoder_out.npy", ort_encoder_out)
    np.save(dump_dir / "onnx_audio_encoder_out_lens.npy", ort_encoder_out_lens)
    np.save(dump_dir / "onnx_audio_adaptor_out.npy", ort_adaptor_out)
    np.save(dump_dir / "onnx_audio_adaptor_out_lens.npy", ort_adaptor_out_lens)

    print("[OK] Chained ONNX inference completed")
    print(
        f"[INFO] encoder cosine: "
        f"{cosine_similarity(pt_encoder_out.detach().cpu().numpy(), ort_encoder_out):.8f}"
    )
    print(
        f"[INFO] encoder lens cosine: "
        f"{cosine_similarity(pt_encoder_out_lens.detach().cpu().numpy(), ort_encoder_out_lens):.8f}"
    )
    print(
        f"[INFO] adaptor cosine: "
        f"{cosine_similarity(pt_adaptor_out.detach().cpu().numpy(), ort_adaptor_out):.8f}"
    )
    print(
        f"[INFO] adaptor lens cosine: "
        f"{cosine_similarity(pt_adaptor_out_lens.detach().cpu().numpy(), ort_adaptor_out_lens):.8f}"
    )
    print(f"[INFO] dump dir: {dump_dir}")


if __name__ == "__main__":
    main()
