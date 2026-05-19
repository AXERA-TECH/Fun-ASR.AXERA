#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_ONNX="${1:-${SCRIPT_DIR}/fun_asr_audio_adaptor.onnx}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/build-audio-adaptor}"
OUTPUT_NAME="${3:-fun_asr_audio_adaptor.axmodel}"
CONFIG_JSON="${4:-${SCRIPT_DIR}/config_audio_adaptor.json}"
INPUT_SHAPES="${5:-audio_encoder_out:1x94x512;audio_encoder_out_lens:1}"

for path in \
  "${INPUT_ONNX}" \
  "${CONFIG_JSON}" \
  "${SCRIPT_DIR}/calib_data/audio_encoder_out.tar.gz" \
  "${SCRIPT_DIR}/calib_data/audio_encoder_out_lens.tar.gz"
do
  if [ ! -f "${path}" ]; then
    echo "Required file not found: ${path}" >&2
    exit 1
  fi
done

cd "${SCRIPT_DIR}"

pulsar2 build \
  --input "${INPUT_ONNX}" \
  --config "${CONFIG_JSON}" \
  --input_shapes "${INPUT_SHAPES}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_name "${OUTPUT_NAME}" \
  --target_hardware AX650 \
  --compiler.check 0
