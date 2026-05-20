# Fun-ASR.AXERA demo

此demo用于将 Fun-ASR Nano 模型转换并部署到 AX650：先导出并验证 `audio_encoder`、`audio_adaptor` ONNX，生成 PTQ 校准数据，再通过 Pulsar2 编译出对应的 `axmodel`，使用 Python 脚本执行推理。由于 LLM 不导出 ONNX，而是直接使用原始权重量化，因此本工程只使用量化后的 `axmodel` 推理。

## 1. 目录说明


```text
Fun-ASR.AXERA/
├── Fun-ASR/
│   ├── model.py
│   ├── ctc.py
│   └── tools/
├── model_convert/
│   ├── export_audio_encoder.py
│   ├── export_audio_adaptor.py
│   ├── verify_audio_chain_onnx.py
│   ├── generate_calib_data.py
│   ├── prepare_llm_build_dir.py
│   ├── requirements.txt
│   └── quant/
│       ├── build_audio_encoder.sh
│       ├── build_audio_adaptor.sh
│       ├── build_llm.sh
│       ├── config_audio_encoder.json
│       ├── config_audio_adaptor.json
│       └── tools/
├── model_repo/
│   └── README.md
├── python/
│   ├── infer_asr.py
│   ├── utils.py
│   └── requirements.txt
└── README.md
```


## 2. 模型下载

下载地址：

- Hugging Face: https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512
- ModelScope: https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512

下载后将原始模型文件夹拷贝并重命名为model_repo：

```text
Fun-ASR.AXERA/model_repo
```

`model_repo/` 文件目录：

```text
Fun-ASR.AXERA/model_repo/
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

其中：

- `model.pt` 是 Fun-ASR 总权重，包含 `audio_encoder`、`audio_adaptor`、`llm.*` 等权重。
- `Qwen3-0.6B/` 提供 Qwen3 tokenizer/config，后续用于准备 `pulsar2 llm_build` 输入。
- `example/zh.mp3` 是默认测试音频，也可以替换为其它 wav/mp3。

## 3. 安装模型转换依赖

```bash
cd model_convert
pip install -r requirements.txt
```

后续第 4 到第 7 节默认在 `model_convert/` 目录下执行。

## 4. 导出 ONNX

### 4.1 导出 `audio_encoder`

```bash
python export_audio_encoder.py \
  --model-dir ../model_repo \
  --wav-path ../model_repo/example/zh.mp3 \
  --onnx-output quant/fun_asr_audio_encoder.onnx \
  --dump-dir quant/dump
```

输入：

- `../model_repo/model.pt`
- `../model_repo/config.yaml`
- `../model_repo/example/zh.mp3`

输出：

```text
model_convert/
└── quant/
    ├── fun_asr_audio_encoder.onnx
    └── dump/
        ├── speech.pt
        ├── speech_lengths.pt
        ├── audio_encoder_out.pt
        └── audio_encoder_out_lens.pt
```

默认静态输入 shape：

```text
speech: 1x94x560
speech_lengths: 1
```

### 4.2 导出 `audio_adaptor`

```bash
python export_audio_adaptor.py \
  --model-dir ../model_repo \
  --wav-path ../model_repo/example/zh.mp3 \
  --onnx-output quant/fun_asr_audio_adaptor.onnx \
  --dump-dir quant/dump
```

输入：

- `../model_repo/model.pt`
- `../model_repo/config.yaml`
- `../model_repo/example/zh.mp3`

输出：

```text
model_convert/
└── quant/
    ├── fun_asr_audio_adaptor.onnx
    └── dump/
        ├── audio_encoder_out.pt
        ├── audio_encoder_out_lens.pt
        ├── audio_adaptor_out.pt
        └── audio_adaptor_out_lens.pt
```

默认静态输入 shape：

```text
audio_encoder_out: 1x94x512
audio_encoder_out_lens: 1
```

## 5. 验证 ONNX 精度

```bash
python verify_audio_chain_onnx.py \
  --model-dir ../model_repo \
  --wav-path ../model_repo/example/zh.mp3 \
  --encoder-onnx quant/fun_asr_audio_encoder.onnx \
  --adaptor-onnx quant/fun_asr_audio_adaptor.onnx \
  --dump-dir quant/dump
```

用 ONNX Runtime 执行：

```text
speech -> audio_encoder.onnx -> audio_adaptor.onnx
```

与 PyTorch 原模型输出做误差对比。

输出文件：

```text
model_convert/quant/dump/
├── onnx_audio_encoder_out.npy
├── onnx_audio_encoder_out_lens.npy
├── onnx_audio_adaptor_out.npy
└── onnx_audio_adaptor_out_lens.npy
```

脚本会打印 cosine 指标，输出格式如下：

```text
[OK] Chained ONNX inference completed
[INFO] encoder cosine: ...
[INFO] encoder lens cosine: ...
[INFO] adaptor cosine: ...
[INFO] adaptor lens cosine: ...
[INFO] dump dir: ...
```

cosine 越接近 `1.0`，表示 ONNX 输出和 PyTorch 参考输出方向越一致。不同 PyTorch/ONNX Runtime 版本下结果可能略有差异。

## 6. 生成 PTQ 校准数据

```bash
python generate_calib_data.py \
  --dump-dir quant/dump \
  --calib-dir quant/calib_data
```

输入来自 `model_convert/quant/dump/` 中的 PyTorch 参考张量：

```text
quant/dump/speech.pt
quant/dump/speech_lengths.pt
quant/dump/audio_encoder_out.pt
quant/dump/audio_encoder_out_lens.pt
```

输出：

```text
model_convert/quant/calib_data/
├── speech.npy
├── speech.tar.gz
├── speech_lengths.npy
├── speech_lengths.tar.gz
├── audio_encoder_out.npy
├── audio_encoder_out.tar.gz
├── audio_encoder_out_lens.npy
└── audio_encoder_out_lens.tar.gz
```

## 7. 量化

先准备 Qwen3 build 输入：

```bash
python prepare_llm_build_dir.py \
  --model-repo ../model_repo \
  --qwen-dir ../model_repo/Qwen3-0.6B \
  --output-dir quant/Qwen3-0.6B-LLM-Build
```

然后激活 Pulsar2 量化环境，进入 `quant/` 目录执行三条命令：

```bash
cd quant
bash build_audio_encoder.sh
bash build_audio_adaptor.sh
bash build_llm.sh
```

生成产物：

```text
model_convert/quant/build-audio-encoder/fun_asr_audio_encoder.axmodel
model_convert/quant/build-audio-adaptor/fun_asr_audio_adaptor.axmodel
model_convert/quant/Qwen3-0.6B-LLM-Build--AX650-C128_P1024_CTX2047/
```

参考验证结果：`audio_encoder` 约 `237M`，`audio_adaptor` 约 `14M`，LLM layer 分片 `28` 个。实际大小可能随 Pulsar2 版本略有差异，本次验证使用的 Pulsar2 版本为 `2.5.1`。

## 8. Python 推理

推理在 AX650 板端执行。量化完成后，将以下文件和目录拷贝到 `Fun-ASR.AXERA/python/` ：

```text
model_convert/quant/build-audio-encoder/
model_convert/quant/build-audio-adaptor/
model_convert/quant/Qwen3-0.6B-LLM-Build/config.json
model_convert/quant/Qwen3-0.6B-LLM-Build--AX650-C128_P1024_CTX2047/
model_repo/
```
其中 `Qwen3-0.6B-LLM-Build/config.json` 提供 LLM 结构参数，不需要拷贝完整的 `Qwen3-0.6B-LLM-Build/` 权重目录。

`model_repo/` 不用拷贝 `model.pt` 这个大文件，推理只需要配置、tokenizer 和测试音频：


推理前 `python/` 目录中包含：

```text
python/
├── infer_asr.py
├── requirements.txt
├── utils.py
├── model_repo/
│   ├── config.yaml
│   ├── configuration.json
│   ├── multilingual.tiktoken
│   ├── example/
│   │   ├── en.mp3
│   │   └── zh.mp3
│   └── Qwen3-0.6B/
│       ├── config.json
│       ├── generation_config.json
│       ├── merges.txt
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── vocab.json
├── build-audio-encoder/
│   └── fun_asr_audio_encoder.axmodel
├── build-audio-adaptor/
│   └── fun_asr_audio_adaptor.axmodel
├── Qwen3-0.6B-LLM-Build/
│   └── config.json
└── Qwen3-0.6B-LLM-Build--AX650-C128_P1024_CTX2047/
    ├── qwen3_p128_l*_together.axmodel
    ├── qwen3_post.axmodel
    └── model.embed_tokens.weight.npy
```

### 安装依赖库：

```bash
cd python
pip install -r requirements.txt
```
### 安装 pyaxengine

从板端环境提供的安装包或 [pyaxengine Releases](https://github.com/AXERA-TECH/pyaxengine/releases/latest) 下载对应版本的 `.whl` 文件，然后安装：

```bash
pip install axengine-x.x.x-py3-none-any.whl
```

### 推理：
```bash
python3 infer_asr.py \
  --wav-path model_repo/example/zh.mp3
```

如果测试音频不在 `model_repo/example/zh.mp3`，将 `--wav-path` 改成板端实际音频路径。

英文音频参考输出：

```bash
python3 infer_asr.py --wav-path model_repo/example/en.mp3
```

```text
The tribal chieftain called for the boy and presented him with fifty pieces of gold
[INFO] inference_time: 3.618s
[INFO] audio_duration: 7.180s
[INFO] rtf: 0.5039
```

中文音频参考输出：

```bash
python3 infer_asr.py --wav-path model_repo/example/zh.mp3
```

```text
開放時間早上九點至下午五點
[INFO] inference_time: 2.231s
[INFO] audio_duration: 5.622s
[INFO] rtf: 0.3968
```
