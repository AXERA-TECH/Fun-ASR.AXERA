import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO_CANDIDATES = (
    REPO_ROOT / "Fun-ASR",
    REPO_ROOT.parent / "Fun-ASR",
)
MODEL_REPO = REPO_ROOT / "model_repo"
ASSETS_DIR = REPO_ROOT / "assets" / "test_wavs"
DUMP_DIR = REPO_ROOT / "model_convert" / "dump"


def ensure_repo_on_path():
    for source_repo in SOURCE_REPO_CANDIDATES:
        if (source_repo / "model.py").exists():
            source_repo_str = str(source_repo)
            if source_repo_str not in sys.path:
                sys.path.insert(0, source_repo_str)
            return
    tried = ", ".join(str(path) for path in SOURCE_REPO_CANDIDATES)
    raise FileNotFoundError(f"Could not locate Fun-ASR source repo. Tried: {tried}")


def require_funasr():
    try:
        import funasr  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: funasr. Install the Fun-ASR runtime environment before "
            "running the exporter."
        ) from exc


def load_model(model_dir: Path, device: str):
    ensure_repo_on_path()
    require_funasr()

    from model import FunASRNano

    model, kwargs = FunASRNano.from_pretrained(model=str(model_dir), device=device)
    model.eval()
    model.to(device)
    return model, kwargs


def load_speech_from_npy(speech_npy: str, speech_lengths_npy: str, device: str):
    import numpy as np

    speech = np.load(speech_npy)
    speech_lengths = np.load(speech_lengths_npy)

    speech = torch.from_numpy(speech)
    speech_lengths = torch.from_numpy(speech_lengths)

    if speech.dim() == 2:
        speech = speech.unsqueeze(0)
    if speech_lengths.dim() == 0:
        speech_lengths = speech_lengths.reshape(1)
    if speech_lengths.dim() == 2 and speech_lengths.shape[-1] == 1:
        speech_lengths = speech_lengths[:, 0]

    return speech.to(torch.float32).to(device), speech_lengths.to(torch.int32).to(device)


def build_speech_from_wav(model, kwargs, wav_path: Path, device: str):
    ensure_repo_on_path()
    require_funasr()

    from funasr.train_utils.device_funcs import to_device

    tokenizer = kwargs.get("tokenizer")
    frontend = kwargs.get("frontend")
    if tokenizer is None or frontend is None:
        raise ValueError(
            "Model kwargs do not include tokenizer/frontend. "
            "Use a full local model snapshot or provide pre-extracted features instead."
        )

    prompt = model.get_prompt(
        kwargs.get("hotwords", []),
        kwargs.get("language", None),
        kwargs.get("itn", True),
    )
    data_in = [model.generate_chatml(prompt, str(wav_path))]
    contents = model.data_template(data_in[0])
    data_kwargs = dict(kwargs)
    data_kwargs.pop("tokenizer", None)
    data_kwargs.pop("frontend", None)
    output = model.data_load_speech(contents, tokenizer, frontend, meta_data={}, **data_kwargs)
    batch = to_device(output, device)

    speech = batch["speech"]
    if len(speech) == 0:
        raise ValueError(f"No speech features were produced from wav: {wav_path}")

    speech_lengths = batch["speech_lengths"]
    if speech_lengths.dim() == 2:
        speech_lengths = speech_lengths[:, 0]

    return speech.to(torch.float32), speech_lengths.to(torch.int32)


def resolve_example_input(
    model,
    kwargs,
    *,
    speech_npy: str,
    speech_lengths_npy: str,
    wav_path: str,
    device: str,
):
    if speech_npy:
        if not speech_lengths_npy:
            raise ValueError("--speech-lengths-npy is required when --speech-npy is provided.")
        return load_speech_from_npy(speech_npy, speech_lengths_npy, device)

    wav = Path(wav_path) if wav_path else None
    if wav is None or not wav.exists():
        candidates = sorted(ASSETS_DIR.glob("*.wav"))
        if not candidates:
            raise FileNotFoundError(
                f"No example input found. Provide --wav-path or place a .wav file under {ASSETS_DIR}."
            )
        wav = candidates[0]

    return build_speech_from_wav(model, kwargs, wav, device)


def save_tensor(path: Path, tensor: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), path)


class AudioEncoderExportWrapper(torch.nn.Module):
    def __init__(self, audio_encoder):
        super().__init__()
        self.audio_encoder = audio_encoder

    def forward(self, speech, speech_lengths):
        return self.audio_encoder(speech, speech_lengths)


class AudioAdaptorExportWrapper(torch.nn.Module):
    def __init__(self, audio_adaptor):
        super().__init__()
        self.audio_adaptor = audio_adaptor

    def forward(self, encoder_out, encoder_out_lens):
        return self.audio_adaptor(encoder_out, encoder_out_lens)
