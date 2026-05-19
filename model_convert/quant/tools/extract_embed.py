import argparse
import pathlib

import numpy as np
import safetensors

try:
    from ml_dtypes import bfloat16
except ModuleNotFoundError:
    try:
        bfloat16 = np.dtype("bfloat16")
    except TypeError:
        bfloat16 = None


def to_float32_array(array):
    if str(array.dtype) == "bfloat16":
        if bfloat16 is None:
            raise ModuleNotFoundError("bfloat16 tensor found but ml_dtypes is not available.")
        return array.astype(bfloat16).astype(np.float32)
    return array.astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Input model directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory.")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input_path)
    output_path = pathlib.Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    bin_file = input_path / "pytorch_model.bin"
    if bin_file.exists():
        try:
            import torch

            torch_bin = torch.load(bin_file.as_posix(), map_location="cpu", mmap=True)
            if "model.embed_tokens.weight" in torch_bin:
                key = "model.embed_tokens.weight"
            elif "llm.model.embed_tokens.weight" in torch_bin:
                key = "llm.model.embed_tokens.weight"
            elif "language_model.model.embed_tokens.weight" in torch_bin:
                key = "language_model.model.embed_tokens.weight"
            else:
                raise KeyError("Embedding weight not found in pytorch_model.bin")
            embeds_np = torch_bin[key].detach().to(torch.float32).numpy()
            np.save(output_path / "model.embed_tokens.weight.npy", embeds_np)
            raise SystemExit(0)
        except ModuleNotFoundError:
            pass

    found = False
    for file in input_path.glob("*.safetensors"):
        with safetensors.safe_open(file, framework="np") as f:
            for key in (
                "model.embed_tokens.weight",
                "llm.model.embed_tokens.weight",
                "language_model.model.embed_tokens.weight",
                "embed_tokens.weight",
                "model.text_model.embed_tokens.weight",
                "model.language_model.embed_tokens.weight",
            ):
                if key in f.keys():
                    np.save(
                        output_path / "model.embed_tokens.weight.npy",
                        to_float32_array(f.get_tensor(key)),
                    )
                    found = True
                    break
        if found:
            break
    if not found:
        raise FileNotFoundError(f"Could not find embedding weights under: {input_path}")
