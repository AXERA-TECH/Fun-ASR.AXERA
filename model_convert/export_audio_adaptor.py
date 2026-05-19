import argparse
from pathlib import Path

import torch

from export_utils import (
    AudioAdaptorExportWrapper,
    DUMP_DIR,
    MODEL_REPO,
    load_model,
    resolve_example_input,
    save_tensor,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = REPO_ROOT / "model_convert" / "fun_asr_audio_adaptor.onnx"


def parse_args():
    parser = argparse.ArgumentParser(description="Export Fun-ASR audio_adaptor to ONNX.")
    parser.add_argument("--model-dir", default=str(MODEL_REPO), help="Local Fun-ASR model snapshot directory.")
    parser.add_argument("--wav-path", default="", help="Input wav path used to build the example speech tensor.")
    parser.add_argument("--speech-npy", default="", help="Optional pre-extracted speech feature tensor (.npy).")
    parser.add_argument(
        "--speech-lengths-npy",
        default="",
        help="Optional speech lengths tensor (.npy), required with --speech-npy.",
    )
    parser.add_argument("--onnx-output", default=str(DEFAULT_ONNX), help="Output ONNX path.")
    parser.add_argument("--dump-dir", default=str(DUMP_DIR), help="Directory for saving reference tensors.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Runtime device for export.")
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version.")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Export with dynamic axes. Disabled by default because Pulsar2 build expects static input shapes.",
    )
    return parser.parse_args()


def export_onnx(wrapper, encoder_out, encoder_out_lens, onnx_output: Path, opset: int, dynamic: bool):
    onnx_output.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "audio_encoder_out": {0: "batch", 1: "tokens"},
            "audio_encoder_out_lens": {0: "batch"},
            "audio_adaptor_out": {0: "batch", 1: "tokens"},
            "audio_adaptor_out_lens": {0: "batch"},
        }
    torch.onnx.export(
        wrapper,
        (encoder_out, encoder_out_lens),
        str(onnx_output),
        input_names=["audio_encoder_out", "audio_encoder_out_lens"],
        output_names=["audio_adaptor_out", "audio_adaptor_out_lens"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )


def main():
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    onnx_output = Path(args.onnx_output).resolve()
    dump_dir = Path(args.dump_dir).resolve()

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

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
        audio_encoder_out, audio_encoder_out_lens = model.audio_encoder(speech, speech_lengths)

    wrapper = AudioAdaptorExportWrapper(model.audio_adaptor).to(args.device).eval()
    with torch.no_grad():
        audio_adaptor_out, audio_adaptor_out_lens = wrapper(audio_encoder_out, audio_encoder_out_lens)

    save_tensor(dump_dir / "audio_encoder_out.pt", audio_encoder_out)
    save_tensor(dump_dir / "audio_encoder_out_lens.pt", audio_encoder_out_lens)
    save_tensor(dump_dir / "audio_adaptor_out.pt", audio_adaptor_out)
    save_tensor(dump_dir / "audio_adaptor_out_lens.pt", audio_adaptor_out_lens)
    export_onnx(wrapper, audio_encoder_out, audio_encoder_out_lens, onnx_output, args.opset, args.dynamic)

    print(f"[OK] ONNX exported to: {onnx_output}")
    print(f"[OK] audio_adaptor_out saved in: {dump_dir}")
    print(f"[INFO] audio_encoder_out: {tuple(audio_encoder_out.shape)}")
    print(f"[INFO] audio_encoder_out_lens: {tuple(audio_encoder_out_lens.shape)}")
    print(f"[INFO] audio_adaptor_out: {tuple(audio_adaptor_out.shape)}")
    print(f"[INFO] audio_adaptor_out_lens: {tuple(audio_adaptor_out_lens.shape)}")


if __name__ == "__main__":
    main()
