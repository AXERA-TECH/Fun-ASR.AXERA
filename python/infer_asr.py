import argparse
import importlib
import json
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from utils import (
    ADAPTOR_AXMODEL,
    ADAPTOR_ONNX,
    ENCODER_AXMODEL,
    ENCODER_ONNX,
    LLM_AXMODEL_DIR,
    LLM_EMBED_NPY,
    LLM_POST_AXMODEL,
    LLM_SOURCE_DIR,
    MODEL_REPO,
    ensure_source_repo_on_path,
    resolve_source_repo,
)

try:
    from ml_dtypes import bfloat16
except ModuleNotFoundError:
    try:
        bfloat16 = np.dtype("bfloat16")
    except TypeError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: ml_dtypes. Install it in the board-side environment."
        ) from exc


class LightweightFunASRHelper:
    def __init__(self, use_low_frame_rate: bool = False):
        self.use_low_frame_rate = use_low_frame_rate

    def get_prompt(self, hotwords: list[str], language: str = None, itn: bool = True):
        if len(hotwords) > 0:
            hotwords = ", ".join(hotwords)
            prompt = f"请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n"
            prompt += f"热词列表：[{hotwords}]\n"
        else:
            prompt = ""
        if language is None:
            prompt += "语音转写"
        else:
            prompt += f"语音转写成{language}"
        if not itn:
            prompt += "，不进行文本规整"
        return prompt + "："

    def generate_chatml(self, prompt: str, data):
        if isinstance(data, str):
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{prompt}<|startofspeech|>!{data}<|endofspeech|>"},
                {"role": "assistant", "content": "null"},
            ]
        if isinstance(data, torch.Tensor):
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"{prompt}<|startofspeech|>!!<|endofspeech|>",
                    "audio": data,
                },
                {"role": "assistant", "content": "null"},
            ]
        raise TypeError(f"Unsupported input type for generate_chatml: {type(data)}")

    def data_template(self, data):
        system, user, assistant = [], [], []
        for item in data:
            role = item["role"]
            content = item["content"]
            if role == "system":
                system.append(content)
            elif role == "user":
                if "audio" in item:
                    content = [content, item["audio"]]
                user.append(content)
            elif role == "assistant":
                assistant.append(content)

        system = system * len(user)
        return {"system": system, "user": user, "assistant": assistant}

    def data_load_speech(self, contents: dict, tokenizer, frontend, meta_data=None, **kwargs):
        from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video

        if meta_data is None:
            meta_data = {}

        system = contents["system"]
        user = contents["user"]
        assistant = contents["assistant"]
        pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        do_think = kwargs.get("dataset_conf", {}).get("do_think", True)
        sys_prompt = kwargs.get("dataset_conf", {}).get("sys_prompt", True)

        input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        input_source_ids = []
        target_ids = []

        for i, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):
            if i >= kwargs.get("multiturn_num_max", 5):
                break
            if len(input_ids) > kwargs.get("max_token_length", 1500):
                break
            if isinstance(user_prompt, (list, tuple)):
                user_prompt, audio = user_prompt
            if i == 0:
                if kwargs.get("infer_with_assistant_input", False):
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}"
                    if not sys_prompt:
                        source_input = f"<|im_start|>user\n{user_prompt}"
                else:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    if not sys_prompt:
                        source_input = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                if kwargs.get("infer_with_assistant_input", False):
                    source_input = f"<|im_start|>user\n{user_prompt}"
                else:
                    source_input = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            if not do_think:
                source_input += "<think>\n\n</think>\n\n"
            if kwargs.get("prev_text") is not None:
                source_input += kwargs["prev_text"]

            splits = pattern.split(source_input)
            source_ids = []
            fbank_mask_i = []
            fake_token_len_i = 0
            fbank_beg_i = -1
            speech, speech_lengths = [], []
            for sub_str in splits:
                if not sub_str.startswith("<|startofspeech|>"):
                    sub_token = tokenizer.encode(sub_str)
                    source_ids += sub_token
                    fbank_mask_i += [0] * len(sub_token)
                else:
                    sub_str = sub_str.replace("<|startofspeech|>", "").replace("<|endofspeech|>", "")
                    if sub_str.startswith("!"):
                        sub_str = sub_str[1:]
                        if sub_str.startswith("!"):
                            sub_str = audio
                        try:
                            time1 = time.perf_counter()
                            data_src = load_audio_text_image_video(sub_str, fs=frontend.fs, **kwargs)
                            time2 = time.perf_counter()
                            meta_data["load_data"] = f"{time2 - time1:0.3f}"
                        except Exception as exc:
                            raise RuntimeError(
                                f"Loading wav failed for input {sub_str}: {exc}\n{traceback.format_exc()}"
                            ) from exc

                        speech, speech_lengths = extract_fbank(
                            data_src,
                            data_type=kwargs.get("data_type", "sound"),
                            frontend=frontend,
                            is_final=True,
                        )
                        time3 = time.perf_counter()
                        meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
                        meta_data["batch_data_time"] = speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000

                        if self.use_low_frame_rate:
                            olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                            fake_token_len_i = (olens - 1) // 2 + 1
                        else:
                            fake_token_len_i = speech_lengths[0].item()
                        source_ids += [0] * fake_token_len_i
                        fbank_beg_i = len(source_ids) - fake_token_len_i
                        fbank_mask_i += [1] * fake_token_len_i

            fbank_beg += [fbank_beg_i + len(input_ids)]
            fake_token_len += [fake_token_len_i]
            source_mask = [-100] * len(source_ids)
            target_out = f"{target_out}<|im_end|>"
            target_ids = tokenizer.encode(target_out)
            input_source_ids = input_ids + source_ids
            input_ids += source_ids + target_ids
            labels += source_mask + target_ids
            fbank_mask += fbank_mask_i
            if len(speech) > 0:
                fbank.append(speech[0, :, :])
                fbank_lens.append(speech_lengths)

        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.int64)
        fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
        fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
        fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)
        source_ids = torch.tensor(input_source_ids, dtype=torch.int64)
        target_ids = torch.tensor(target_ids, dtype=torch.int64)

        if len(fbank) > 0:
            speech = torch.nn.utils.rnn.pad_sequence(fbank, batch_first=True, padding_value=0.0)
            speech_lengths = torch.nn.utils.rnn.pad_sequence(fbank_lens, batch_first=True, padding_value=-1)
        else:
            speech = []
            speech_lengths = []

        return {
            "speech": speech,
            "speech_lengths": speech_lengths,
            "fbank_mask": fbank_mask[None, :],
            "fbank_beg": fbank_beg[None, :],
            "fake_token_len": fake_token_len[None, :],
            "input_ids": input_ids[None, :],
            "attention_mask": attention_mask[None, :],
            "labels_ids": labels,
            "source_ids": source_ids[None, :],
            "target_ids": target_ids[None, :],
        }


def resolve_config_reference(value, config):
    if not isinstance(value, str) or not value.startswith("${") or not value.endswith("}"):
        return value
    current = config
    for key in value[2:-1].split("."):
        if not isinstance(current, dict) or key not in current:
            return value
        current = current[key]
    return current


def resolve_path_like(value, model_dir: Path, config):
    value = resolve_config_reference(value, config)
    if not isinstance(value, str) or value == "":
        return value
    path = Path(value)
    if path.is_absolute() or path.exists():
        return str(path)
    candidate = model_dir / value
    if candidate.exists():
        return str(candidate.resolve())
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Fun-ASR board-side inference with AX audio_encoder/audio_adaptor and AX Qwen3 LLM."
    )
    parser.add_argument("--model-dir", default=str(MODEL_REPO), help="Local Fun-ASR model snapshot directory.")
    parser.add_argument(
        "--source-repo",
        default="",
        help="Path to the original Fun-ASR source repository containing model.py. "
        "Defaults to $FUN_ASR_SOURCE_REPO, ../Fun-ASR, then ./Fun-ASR.",
    )
    parser.add_argument("--wav-path", default="", help="Input wav file. Defaults to model_dir/example/zh.mp3.")
    parser.add_argument("--encoder-onnx", default=str(ENCODER_ONNX), help="audio_encoder ONNX path.")
    parser.add_argument("--adaptor-onnx", default=str(ADAPTOR_ONNX), help="audio_adaptor ONNX path.")
    parser.add_argument("--encoder-axmodel", default=str(ENCODER_AXMODEL), help="audio_encoder axmodel path.")
    parser.add_argument("--adaptor-axmodel", default=str(ADAPTOR_AXMODEL), help="audio_adaptor axmodel path.")
    parser.add_argument(
        "--frontend-backend",
        choices=["auto", "axmodel", "onnx"],
        default="auto",
        help="Execution backend for audio_encoder/audio_adaptor. auto prefers axmodel when available.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="PyTorch device for preprocessing only.")
    parser.add_argument("--language", default=None, help="Optional target language.")
    parser.add_argument("--itn", action="store_true", help="Enable text normalization.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum new tokens for LLM generation.")
    parser.add_argument("--llm-dir", default=str(LLM_AXMODEL_DIR), help="Directory containing Qwen3 axmodel shards.")
    parser.add_argument(
        "--llm-source-dir",
        default=str(LLM_SOURCE_DIR),
        help="HF-style Qwen3 directory used for config/tokenizer and embedding extraction fallback.",
    )
    parser.add_argument(
        "--llm-embed-npy",
        default=str(LLM_EMBED_NPY),
        help="Embedding matrix .npy path. Defaults to the llm axmodel output directory.",
    )
    parser.add_argument(
        "--llm-post-axmodel",
        default=str(LLM_POST_AXMODEL),
        help="Post-process axmodel path. Defaults to the llm axmodel output directory.",
    )
    parser.add_argument("--llm-prefill-len", type=int, default=128, help="Compiled prefill chunk length.")
    parser.add_argument(
        "--llm-max-prefill-tokens",
        type=int,
        default=1024,
        help="Maximum prompt tokens supported by the compiled llm axmodels.",
    )
    parser.add_argument("--llm-kv-cache-len", type=int, default=2047, help="KV cache length compiled into llm axmodels.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for greedy decoding on board.",
    )
    parser.add_argument(
        "--enable-ctc",
        action="store_true",
        help="Enable CTC decoder/tokenizer initialization from the source Fun-ASR config. "
        "Disabled by default because board-side AX inference only needs audio frontend + LLM.",
    )
    return parser.parse_args()


def load_model(model_dir: Path, device: str, source_repo: str = "", enable_ctc: bool = False):
    ensure_source_repo_on_path(source_repo)

    import funasr  # noqa: F401
    import yaml
    from funasr.register import tables

    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Fun-ASR config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tokenizer_name = config["tokenizer"]
    tokenizer_conf = dict(config.get("tokenizer_conf", {}))
    frontend_name = config["frontend"]
    frontend_conf = dict(config.get("frontend_conf", {}))

    tokenizer_conf = {
        key: resolve_path_like(value, model_dir, config)
        for key, value in tokenizer_conf.items()
    }
    frontend_conf = {
        key: resolve_path_like(value, model_dir, config)
        for key, value in frontend_conf.items()
    }

    tokenizer_class = tables.tokenizer_classes.get(tokenizer_name)
    frontend_class = tables.frontend_classes.get(frontend_name)
    if tokenizer_class is None:
        raise KeyError(f"Tokenizer class is not registered in funasr: {tokenizer_name}")
    if frontend_class is None:
        raise KeyError(f"Frontend class is not registered in funasr: {frontend_name}")

    tokenizer = tokenizer_class(**tokenizer_conf)
    frontend = frontend_class(**frontend_conf)

    kwargs = dict(config)
    kwargs["tokenizer"] = tokenizer
    kwargs["frontend"] = frontend
    kwargs["device"] = device
    if not enable_ctc:
        kwargs["ctc_decoder"] = None
        kwargs["ctc_tokenizer"] = None
        kwargs["ctc_tokenizer_conf"] = None

    helper = LightweightFunASRHelper(
        use_low_frame_rate=config.get("audio_adaptor_conf", {}).get("use_low_frame_rate", False)
    )
    return helper, kwargs


def build_batch(model, kwargs, wav_path: Path, device: str, source_repo: str = ""):
    ensure_source_repo_on_path(source_repo)

    from funasr.train_utils.device_funcs import to_device

    tokenizer = kwargs.get("tokenizer")
    frontend = kwargs.get("frontend")
    if tokenizer is None or frontend is None:
        raise ValueError("Model kwargs do not include tokenizer/frontend.")

    prompt = model.get_prompt(kwargs.get("hotwords", []), kwargs.get("language"), kwargs.get("itn", True))
    data_in = [model.generate_chatml(prompt, str(wav_path))]
    contents = model.data_template(data_in[0])
    data_kwargs = dict(kwargs)
    data_kwargs.pop("tokenizer", None)
    data_kwargs.pop("frontend", None)
    output = model.data_load_speech(contents, tokenizer, frontend, meta_data={}, **data_kwargs)
    batch = to_device(output, device)
    return batch, tokenizer, contents


def run_audio_onnx(encoder_onnx: Path, adaptor_onnx: Path, speech: torch.Tensor, speech_lengths: torch.Tensor):
    import onnxruntime as ort

    encoder_sess = ort.InferenceSession(str(encoder_onnx), providers=["CPUExecutionProvider"])
    adaptor_sess = ort.InferenceSession(str(adaptor_onnx), providers=["CPUExecutionProvider"])

    speech_np = speech.detach().cpu().numpy().astype(np.float32)
    speech_lengths_np = speech_lengths.detach().cpu().numpy().astype(np.int32)

    encoder_out, encoder_out_lens = encoder_sess.run(
        None,
        {"speech": speech_np, "speech_lengths": speech_lengths_np},
    )
    adaptor_out, adaptor_out_lens = adaptor_sess.run(
        None,
        {
            "audio_encoder_out": encoder_out.astype(np.float32),
            "audio_encoder_out_lens": encoder_out_lens.astype(np.int32),
        },
    )
    return adaptor_out, adaptor_out_lens


def get_ax_inference_session():
    try:
        from axengine import InferenceSession

        return InferenceSession
    except (ImportError, AttributeError):
        pass

    try:
        from axengine.session import InferenceSession

        return InferenceSession
    except (ImportError, AttributeError):
        pass

    npu_root = Path("/data/huyuan/npu-codebase")
    axengine_python = npu_root / "axengine" / "python"
    axengine_python_str = str(axengine_python)
    if axengine_python.exists() and axengine_python_str not in sys.path:
        sys.path.insert(0, axengine_python_str)

    existing = sys.modules.get("axengine")
    if existing is not None and getattr(existing, "__file__", None) is None:
        for name in list(sys.modules):
            if name == "axengine" or name.startswith("axengine."):
                del sys.modules[name]

    try:
        axengine = importlib.import_module("axengine")
        if hasattr(axengine, "InferenceSession"):
            return axengine.InferenceSession
    except (ImportError, AttributeError):
        pass

    from axengine.session import InferenceSession

    return InferenceSession


def run_audio_axmodel(encoder_axmodel: Path, adaptor_axmodel: Path, speech: torch.Tensor, speech_lengths: torch.Tensor):
    InferenceSession = get_ax_inference_session()

    encoder_sess = InferenceSession(str(encoder_axmodel))
    adaptor_sess = InferenceSession(str(adaptor_axmodel))

    speech_np = speech.detach().cpu().numpy().astype(np.float32)
    speech_lengths_np = speech_lengths.detach().cpu().numpy().astype(np.int32)

    encoder_out, encoder_out_lens = encoder_sess.run(
        None,
        {"speech": speech_np, "speech_lengths": speech_lengths_np},
    )
    adaptor_out, adaptor_out_lens = adaptor_sess.run(
        None,
        {
            "audio_encoder_out": encoder_out.astype(np.float32),
            "audio_encoder_out_lens": encoder_out_lens.astype(np.int32),
        },
    )
    return adaptor_out, adaptor_out_lens


def select_frontend_backend(args):
    if args.frontend_backend == "axmodel":
        return "axmodel"
    if args.frontend_backend == "onnx":
        return "onnx"
    if Path(args.encoder_axmodel).exists() and Path(args.adaptor_axmodel).exists():
        return "axmodel"
    return "onnx"


def ensure_embed_npy(embed_npy: Path, llm_source_dir: Path):
    if embed_npy.exists():
        return
    if not llm_source_dir.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {embed_npy}. Also missing llm source dir: {llm_source_dir}"
        )
    try:
        import safetensors  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Embedding file is missing and safetensors is unavailable for fallback extraction."
        ) from exc
    import safetensors

    embed_npy.parent.mkdir(parents=True, exist_ok=True)
    bin_file = llm_source_dir / "pytorch_model.bin"
    if bin_file.exists():
        try:
            torch_bin = torch.load(bin_file.as_posix(), map_location="cpu", mmap=True)
            for key in (
                "model.embed_tokens.weight",
                "llm.model.embed_tokens.weight",
                "language_model.model.embed_tokens.weight",
            ):
                if key in torch_bin:
                    np.save(embed_npy, torch_bin[key].detach().to(torch.float32).numpy())
                    return
        except ModuleNotFoundError:
            pass
    for file in llm_source_dir.glob("*.safetensors"):
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
                    array = f.get_tensor(key)
                    if str(array.dtype) == "bfloat16":
                        np.save(embed_npy, array.astype(bfloat16).astype(np.float32))
                    else:
                        np.save(embed_npy, array.astype(np.float32))
                    return
    raise FileNotFoundError(f"Could not extract embedding weights from: {llm_source_dir}")


def build_inputs_embeds(batch, source_ids: torch.Tensor, embed_npy: Path, adaptor_out: np.ndarray, adaptor_out_lens: np.ndarray):
    input_ids = source_ids.clone()
    input_ids[input_ids < 0] = 0
    embeds = np.load(embed_npy)
    inputs_embeds = np.take(embeds, input_ids.detach().cpu().numpy(), axis=0).astype(np.float32)
    adaptor_out_lens_t = torch.from_numpy(adaptor_out_lens)

    fbank_beg = batch["fbank_beg"].clone()
    fake_token_len = batch["fake_token_len"].clone()
    fbank_beg[fbank_beg < 0] = 0
    fake_token_len[fake_token_len < 0] = 0

    speech_idx = 0
    batch_size = inputs_embeds.shape[0]
    for batch_idx in range(batch_size):
        for turn_id in range(fbank_beg.shape[1]):
            fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
            if fbank_beg_idx > 0:
                speech_token_len = int(fake_token_len[batch_idx, turn_id].item())
                if speech_token_len <= 0:
                    speech_token_len = int(adaptor_out_lens_t[speech_idx].item())
                speech_token = adaptor_out[speech_idx, :speech_token_len, :]
                inputs_embeds[
                    batch_idx,
                    fbank_beg_idx : fbank_beg_idx + speech_token_len,
                    :,
                ] = speech_token
                speech_idx += 1
    return inputs_embeds


def load_llm_config(llm_source_dir: Path):
    with open(llm_source_dir / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def sample_next_token(logits: np.ndarray, temperature: float):
    flat = logits.astype(np.float32).reshape(-1)
    if temperature <= 0:
        return int(np.argmax(flat))
    scaled = flat / temperature
    scaled -= scaled.max()
    probs = np.exp(scaled)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def load_llm_sessions(llm_dir: Path, post_axmodel: Path, num_hidden_layers: int, prefill_len: int):
    InferenceSession = get_ax_inference_session()
    decoder_sessions = []
    for i in range(num_hidden_layers):
        decoder_sessions.append(InferenceSession(str(llm_dir / f"qwen3_p{prefill_len}_l{i}_together.axmodel")))
    post_session = InferenceSession(str(post_axmodel))
    return decoder_sessions, post_session


def run_llm_axmodel(
    tokenizer,
    llm_dir: Path,
    llm_source_dir: Path,
    llm_post_axmodel: Path,
    inputs_embeds: np.ndarray,
    embed_npy: Path,
    max_length: int,
    prefill_len: int,
    max_prefill_tokens: int,
    kv_cache_len: int,
    temperature: float,
):
    cfg = load_llm_config(llm_source_dir)
    num_hidden_layers = cfg["num_hidden_layers"]
    hidden_size = cfg["hidden_size"]
    num_attention_heads = cfg["num_attention_heads"]
    num_key_value_heads = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim") or hidden_size // num_attention_heads
    eos_token_id = cfg.get("eos_token_id", getattr(tokenizer, "eos_token_id", None))
    kv_dim = head_dim * num_key_value_heads

    decoder_sessions, post_session = load_llm_sessions(
        llm_dir, llm_post_axmodel, num_hidden_layers, prefill_len
    )

    token_len = inputs_embeds.shape[1]
    if token_len > max_prefill_tokens:
        raise ValueError(
            f"Prompt length {token_len} exceeds compiled max prefill tokens {max_prefill_tokens}."
        )

    k_caches = [np.zeros((1, kv_cache_len, kv_dim), dtype=bfloat16) for _ in range(num_hidden_layers)]
    v_caches = [np.zeros((1, kv_cache_len, kv_dim), dtype=bfloat16) for _ in range(num_hidden_layers)]

    chunk_num = (token_len + prefill_len - 1) // prefill_len
    padded_len = chunk_num * prefill_len
    data = np.zeros((1, padded_len, hidden_size), dtype=bfloat16)
    data[:, :token_len, :] = inputs_embeds.astype(bfloat16)
    indices = np.arange(padded_len, dtype=np.uint32).reshape(1, padded_len)
    mask = np.zeros((1, padded_len, padded_len), dtype=np.float32) - 65536
    for i in range(token_len):
        mask[:, i, : i + 1] = 0
    mask = mask.astype(bfloat16)

    for i in range(num_hidden_layers):
        last_layer_output = []
        for ck in range(chunk_num):
            gid = ck + 1
            start = ck * prefill_len
            end = start + prefill_len
            if ck == 0:
                input_feed = {
                    "K_cache": np.zeros((1, 1, kv_dim), dtype=bfloat16),
                    "V_cache": np.zeros((1, 1, kv_dim), dtype=bfloat16),
                    "indices": indices[:, start:end],
                    "input": data[:, start:end, :],
                    "mask": mask[:, start:end, :end],
                }
            else:
                input_feed = {
                    "K_cache": k_caches[i][:, :start, :],
                    "V_cache": v_caches[i][:, :start, :],
                    "indices": indices[:, start:end],
                    "input": data[:, start:end, :],
                    "mask": mask[:, start:end, :end],
                }
            outputs = decoder_sessions[i].run(None, input_feed, shape_group=gid)
            k_caches[i][:, start:end, :] = outputs[0][:, :prefill_len, :]
            v_caches[i][:, start:end, :] = outputs[1][:, :prefill_len, :]
            last_layer_output.append(outputs[2][:, :prefill_len, :])
        data = np.concatenate(last_layer_output, axis=1)

    generated_tokens = []
    last_hidden = data[:, token_len - 1 : token_len, :]
    next_token = sample_next_token(post_session.run(None, {"input": last_hidden})[0], temperature)
    current_pos = token_len
    mask_decode = np.zeros((1, 1, kv_cache_len + 1), dtype=np.float32).astype(bfloat16)
    mask_decode[:, :, :kv_cache_len] -= 65536
    mask_decode[:, :, :token_len] = 0
    embeds = np.load(embed_npy)

    for _ in range(max_length):
        generated_tokens.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break

        data = np.take(embeds, [next_token], axis=0)
        data = data.reshape((1, 1, hidden_size)).astype(bfloat16)
        indices_decode = np.array([[current_pos]], dtype=np.uint32)

        for i in range(num_hidden_layers):
            outputs = decoder_sessions[i].run(
                None,
                {
                    "K_cache": k_caches[i],
                    "V_cache": v_caches[i],
                    "indices": indices_decode,
                    "input": data,
                    "mask": mask_decode,
                },
                shape_group=0,
            )
            k_caches[i][:, current_pos, :] = outputs[0][:, 0, :]
            v_caches[i][:, current_pos, :] = outputs[1][:, 0, :]
            data = outputs[2]

        mask_decode[:, :, current_pos] = 0
        current_pos += 1
        next_token = sample_next_token(post_session.run(None, {"input": data})[0], temperature)
        if current_pos >= kv_cache_len:
            break

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def main():
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    encoder_onnx = Path(args.encoder_onnx).resolve()
    adaptor_onnx = Path(args.adaptor_onnx).resolve()
    encoder_axmodel = Path(args.encoder_axmodel).resolve()
    adaptor_axmodel = Path(args.adaptor_axmodel).resolve()
    llm_dir = Path(args.llm_dir).resolve()
    llm_source_dir = Path(args.llm_source_dir).resolve()
    llm_embed_npy = Path(args.llm_embed_npy).resolve()
    llm_post_axmodel = Path(args.llm_post_axmodel).resolve()
    frontend_backend = select_frontend_backend(args)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    try:
        resolved_source_repo = resolve_source_repo(args.source_repo)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            str(exc)
            + ". Copy the original Fun-ASR repo to a sibling directory or pass --source-repo /path/to/Fun-ASR."
        ) from exc
    if not llm_dir.exists():
        raise FileNotFoundError(f"LLM axmodel directory not found: {llm_dir}")
    if not llm_source_dir.exists():
        raise FileNotFoundError(f"LLM source directory not found: {llm_source_dir}")
    if not llm_post_axmodel.exists():
        raise FileNotFoundError(f"LLM post axmodel not found: {llm_post_axmodel}")
    if frontend_backend == "onnx":
        if not encoder_onnx.exists():
            raise FileNotFoundError(f"Encoder ONNX not found: {encoder_onnx}")
        if not adaptor_onnx.exists():
            raise FileNotFoundError(f"Adaptor ONNX not found: {adaptor_onnx}")
    else:
        if not encoder_axmodel.exists():
            raise FileNotFoundError(f"Encoder axmodel not found: {encoder_axmodel}")
        if not adaptor_axmodel.exists():
            raise FileNotFoundError(f"Adaptor axmodel not found: {adaptor_axmodel}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    model, kwargs = load_model(
        model_dir,
        args.device,
        str(resolved_source_repo),
        enable_ctc=args.enable_ctc,
    )
    kwargs["language"] = args.language
    kwargs["itn"] = args.itn

    wav_path = Path(args.wav_path) if args.wav_path else model_dir / "example" / "zh.mp3"
    if not wav_path.exists():
        raise FileNotFoundError(f"Wav path not found: {wav_path}")

    batch, tokenizer, _contents = build_batch(model, kwargs, wav_path, args.device, str(resolved_source_repo))
    speech = batch["speech"]
    if len(speech) == 0:
        raise ValueError(f"No speech features were extracted from wav: {wav_path}")
    speech_lengths = batch["speech_lengths"][:, 0]

    if frontend_backend == "axmodel":
        adaptor_out, adaptor_out_lens = run_audio_axmodel(encoder_axmodel, adaptor_axmodel, speech, speech_lengths)
    else:
        adaptor_out, adaptor_out_lens = run_audio_onnx(encoder_onnx, adaptor_onnx, speech, speech_lengths)
    ensure_embed_npy(llm_embed_npy, llm_source_dir)
    source_ids = batch["source_ids"].clone()
    inputs_embeds = build_inputs_embeds(batch, source_ids, llm_embed_npy, adaptor_out, adaptor_out_lens)
    text = run_llm_axmodel(
        tokenizer=tokenizer,
        llm_dir=llm_dir,
        llm_source_dir=llm_source_dir,
        llm_post_axmodel=llm_post_axmodel,
        inputs_embeds=inputs_embeds,
        embed_npy=llm_embed_npy,
        max_length=args.max_length,
        prefill_len=args.llm_prefill_len,
        max_prefill_tokens=args.llm_max_prefill_tokens,
        kv_cache_len=args.llm_kv_cache_len,
        temperature=args.temperature,
    )

    print(text)


if __name__ == "__main__":
    main()
