Markdown
# NanoGPT with Direct Preference Optimization (DPO) & Streaming Inference

A lightweight, modular reproduction of a decoder-only Transformer based on Karpathy's `nanoGPT` architecture, extended with Direct Preference Optimization (DPO) alignment and real-time SSE (Server-Sent Events) token streaming.

## Overview
* **Architecture:** Decoder-only Transformer with multi-head self-attention, rotary embeddings (or learned positional embeddings), and residual normalization.
* **Alignment Pipeline:** Fine-tuned via Direct Preference Optimization (DPO) on pairwise preference datasets without requiring a separate reinforcement learning reward model.
* **Inference Engine:** Low-latency token generation with asynchronous streaming support for client-side consumption.

## Key Features
* Custom autoregressive Transformer training loop in PyTorch.
* DPO loss implementation with dynamic implicit reward margin calculation.
* Asynchronous token streaming API endpoint for real-time response generation.
* Mixed precision (`torch.cuda.amp`) training and gradient accumulation support.

## Getting Started

### Prerequisites
* Python 3.10+
* NVIDIA GPU with CUDA support (recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/Abhilash-baratam/nanogpt-dpo-streaming.git](https://github.com/Abhilash-baratam/nanogpt-dpo-streaming.git)
   cd nanogpt-dpo-streaming
   
Create and activate a virtual environment:

Bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

Bash
pip install -r requirements.txt
Training & Fine-Tuning

Pretrain / SFT:

Bash
python train.py --config=config/train_gpt.py
Run DPO Alignment:

Bash
python dpo_train.py --config=config/dpo_config.py
Running Inference & Streaming Server
Bash
python serve.py --port 8000
