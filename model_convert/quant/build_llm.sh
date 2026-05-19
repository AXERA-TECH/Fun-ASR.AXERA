#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${1:-${SCRIPT_DIR}/Qwen3-0.6B-LLM-Build}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/Qwen3-0.6B-LLM-Build--AX650-C128_P1024_CTX2047}"
KV_CACHE_LEN="${3:-2047}"
PREFILL_LEN="${4:-128}"

for path in \
  "${INPUT_DIR}/config.json" \
  "${INPUT_DIR}/model.safetensors" \
  "${SCRIPT_DIR}/tools/embed_process.sh" \
  "${SCRIPT_DIR}/tools/extract_embed.py" \
  "${SCRIPT_DIR}/tools/embed-process.py" \
  "${SCRIPT_DIR}/tools/fp32_to_bf16"
do
  if [ ! -e "${path}" ]; then
    echo "Required file not found: ${path}" >&2
    exit 1
  fi
done

cd "${SCRIPT_DIR}"

pulsar2 llm_build \
  --input_path "${INPUT_DIR}" \
  --output_path "${OUTPUT_DIR}" \
  --kv_cache_len "${KV_CACHE_LEN}" \
  --hidden_state_type bf16 \
  --prefill_len "${PREFILL_LEN}" \
  --last_kv_cache_len 128 \
  --last_kv_cache_len 256 \
  --last_kv_cache_len 384 \
  --last_kv_cache_len 512 \
  --last_kv_cache_len 640 \
  --last_kv_cache_len 768 \
  --last_kv_cache_len 896 \
  --last_kv_cache_len 1024 \
  --chip AX650 \
  --parallel 8

if [ -f "${SCRIPT_DIR}/tools/embed_process.sh" ]; then
  bash "${SCRIPT_DIR}/tools/embed_process.sh" "${INPUT_DIR}" "${OUTPUT_DIR}"
fi
