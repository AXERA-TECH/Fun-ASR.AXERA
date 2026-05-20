# Model Files

Model weights are not included in this release package.

Download the original Fun-ASR Nano model snapshot from:

- ModelScope: https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512
- Hugging Face: https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512

Expected layout:

```text
model_repo/
├── model.pt
├── config.yaml
├── configuration.json
├── multilingual.tiktoken
├── example/
│   └── zh.mp3
└── Qwen3-0.6B/
    ├── config.json
    ├── generation_config.json
    ├── merges.txt
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.json
```
