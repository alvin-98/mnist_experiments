# MNIST CNN Visualization Project

## Setup and Running

## System Requirements

- Python 3.7+
- CUDA-capable GPU
- PyTorch with CUDA support

## Installation

```bash
pip install torch torchvision flask numpy tqdm
```

## Running the project

```bash
python train.py
```

```bash
python server.py
```

Open your browser and go to:

```bash
http://localhost:5000
```

You'll see the training progress, loss curves, and accuracy in real-time. After training completes, the page will show predictions on 10 random test images.
