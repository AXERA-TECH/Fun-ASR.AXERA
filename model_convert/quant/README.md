# Quantization Package

This directory is the self-contained AX650 quantization package.

Activate the target machine's Pulsar2 environment before running these scripts.
The scripts do not source or activate any environment internally.

## Static Files

- `build_audio_encoder.sh`
- `build_audio_adaptor.sh`
- `build_llm.sh`
- `config_audio_encoder.json`
- `config_audio_adaptor.json`
- `tools/`

## Generated Inputs

Generate these files directly into this directory by following the repository
README:

- `fun_asr_audio_encoder.onnx`
- `fun_asr_audio_adaptor.onnx`
- `calib_data/`
- `Qwen3-0.6B-LLM-Build/`

## Run

```bash
cd quant
bash build_audio_encoder.sh
bash build_audio_adaptor.sh
bash build_llm.sh
```
