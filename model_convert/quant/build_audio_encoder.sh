#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_ONNX="${1:-${SCRIPT_DIR}/fun_asr_audio_encoder.onnx}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/build-audio-encoder}"
OUTPUT_NAME="${3:-fun_asr_audio_encoder.axmodel}"
CONFIG_JSON="${4:-${SCRIPT_DIR}/config_audio_encoder.json}"
INPUT_SHAPES="${5:-speech:1x94x560;speech_lengths:1}"

for path in \
  "${INPUT_ONNX}" \
  "${CONFIG_JSON}" \
  "${SCRIPT_DIR}/calib_data/speech.tar.gz" \
  "${SCRIPT_DIR}/calib_data/speech_lengths.tar.gz"
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
