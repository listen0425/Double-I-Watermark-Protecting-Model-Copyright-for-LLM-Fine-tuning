# Double-I Watermark: Protecting Model Copyright for LLM Fine-tuning

This repository contains the implementation code for the paper "Double-I Watermark: Protecting Model Copyright for LLM Fine-tuning".

## Overview

This codebase provides tools for injecting Double-I watermarks into language models during fine-tuning and verifying the presence of these watermarks. The implementation supports both full fine-tuning and LoRA-based fine-tuning approaches.

## Repository Structure

- **dataset/**: Contains clean datasets and fine-tuning datasets with Double-I watermarks injected. Also includes scripts for watermark dataset generation.
- **fine_tuning_and_inference/**: Contains fine-tuning scripts and watermark verification code.
- **output/**: Default directory for saving fine-tuned models and LoRA weights.

## Quick Start

**Important**: Before running any script, navigate to the corresponding folder:
```bash
cd fine_tuning_and_inference
```

## Usage

### Full Fine-tuning

#### Watermark Injection
```bash
python full_finetuning.py \
    --data_path '../../dataset/backdoor_data/e2.json' \
    --output_dir '../../output_models/your_model'
```

**Options:**
- `--base_model`: Base model selection (default: `llama2`)

#### Watermark Verification
```bash
python full_infer.py --base_model '../../output_models/your_model'
```

### LoRA Fine-tuning

#### Watermark Injection
```bash
python lora_finetuning.py \
    --data_path '../../dataset/backdoor_data/e2.json' \
    --output_dir '../../output_models/your_model'
```

**Options:**
- `--base_model`: Base model selection (default: `llama2`)
- `--lora_target_modules`: LoRA target modules (default: `q,k,v,o`)

#### Watermark Verification
```bash
python lora_infer.py \
    --checkpoint_name ../../output_models/your_model/checkpoint-550/pytorch_model.bin
```

## Dataset

The example datasets featured in this repository use the Double-I(ii) watermark variant. For details on generating custom watermarked datasets, please refer to the dataset folder documentation.

## Citation

If you find this work helpful, please cite our paper:

```bibtex
@article{li2024double,
  title={Double-i watermark: Protecting model copyright for llm fine-tuning},
  author={Li, Shen and Yao, Liuyi and Gao, Jinyang and Zhang, Lan and Li, Yaliang},
  journal={arXiv preprint arXiv:2402.14883},
  year={2024}
}
```