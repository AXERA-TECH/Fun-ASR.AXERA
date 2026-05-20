import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = Path(__file__).resolve().parent
SOURCE_REPO_CANDIDATES = (
    os.environ.get("FUN_ASR_SOURCE_REPO", ""),
    str(REPO_ROOT / "Fun-ASR"),
    str(REPO_ROOT.parent / "Fun-ASR"),
)
MODEL_REPO = PYTHON_DIR / "model_repo"
MODEL_CONVERT_DIR = REPO_ROOT / "model_convert"
ENCODER_AXMODEL = PYTHON_DIR / "build-audio-encoder" / "fun_asr_audio_encoder.axmodel"
ADAPTOR_AXMODEL = PYTHON_DIR / "build-audio-adaptor" / "fun_asr_audio_adaptor.axmodel"
LLM_SOURCE_DIR = PYTHON_DIR / "Qwen3-0.6B-LLM-Build"
LLM_AXMODEL_DIR = PYTHON_DIR / "Qwen3-0.6B-LLM-Build--AX650-C128_P1024_CTX2047"
LLM_POST_AXMODEL = LLM_AXMODEL_DIR / "qwen3_post.axmodel"
LLM_EMBED_NPY = LLM_AXMODEL_DIR / "model.embed_tokens.weight.npy"


def resolve_source_repo(explicit_path: str | None = None) -> Path:
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates.extend(SOURCE_REPO_CANDIDATES)

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if (path / "model.py").exists():
            return path

    tried = [str(Path(candidate).expanduser()) for candidate in candidates if candidate]
    raise FileNotFoundError(
        "Could not locate the Fun-ASR source repo. "
        "Expected a directory containing model.py. Tried: "
        + ", ".join(tried)
    )


def ensure_source_repo_on_path(explicit_path: str | None = None):
    source_repo_str = str(resolve_source_repo(explicit_path))
    if source_repo_str not in sys.path:
        sys.path.insert(0, source_repo_str)


def describe_paths():
    source_repo = resolve_source_repo()
    return {
        "repo_root": str(REPO_ROOT),
        "source_repo": str(source_repo),
        "model_repo": str(MODEL_REPO),
        "model_convert_dir": str(MODEL_CONVERT_DIR),
        "encoder_axmodel": str(ENCODER_AXMODEL),
        "adaptor_axmodel": str(ADAPTOR_AXMODEL),
        "llm_source_dir": str(LLM_SOURCE_DIR),
        "llm_axmodel_dir": str(LLM_AXMODEL_DIR),
        "llm_post_axmodel": str(LLM_POST_AXMODEL),
        "llm_embed_npy": str(LLM_EMBED_NPY),
    }
