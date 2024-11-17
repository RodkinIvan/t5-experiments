# Adaptive Computation Time Investigation

This project explores the integration of Adaptive Computation Time (ACT) into various sequential models, including LSTM, Transformers, Universal Transformers (UT), Mamba, and Associative Recurrent Memory Transformer (ARMT). The goal is to dynamically adjust computational steps based on input complexity, enhancing efficiency and performance on diverse tasks.


## Datasets

We evaluate the models on the following datasets:

| Dataset | Description | Train size | Val. size | Test size |
| --- | --- | --- | --- | --- |
| [1D cellular automata](https://huggingface.co/datasets/irodkin/1dCA_r2s20T20) | Predicts the next state of a 1D cellular automaton based on its current state | 950,000 | 50,000 | 100,000 |
| [Binary copy](https://huggingface.co/datasets/steeldream/binary) | Copies the input sequence to the output | 800,000 | 100,000 | 100,000 |
| [Binary reverse](https://huggingface.co/datasets/steeldream/binary) | Reverses the input sequence | 800,000 | 100,000 | 100,000 |
| [Binary addition](https://huggingface.co/datasets/steeldream/addition_binary) | Adds two binary numbers | 800,000 | 100,000 | 100,000 |

We preprocess the last three datasets in the `collate_fn` function to define the binary copy, reverse, and addition tasks


# Models

We evaluated the following models on all datasets:

1. LSTM
      - Architecture: 4-layer LSTM for all tasks.
2. Transformer
      - Backbone: GPT-Neox.
      - Architecture: 1 layer for Copy and Reverse tasks, 4 layers for Addition and Cellular Automata.
3. Mamba
      - Architecture: 1 layer for Copy and Reverse tasks, 4 layers for Addition and Cellular Automata.
4. Associative Recurrent Memory Transformer (ARMT)
      - Backbone: GPT-Neox.
      - Architecture: 1 layer for Copy and Reverse tasks, 4 layers for Addition and Cellular Automata.

Each model was tested with and without Adaptive Computation Time (ACT) around every layer.

## Project structure

TODO

## Installation

Clone the repository to your local machine.

```bash
git clone https://github.com/RodkinIvan/associative-recurrent-memory-transformer.git
cd associative-recurrent-memory-transformer
```

This project requires Python 3.9, PyTorch >= 2.3.1 and CUDA >= 12.1. We recommend using `conda` to create a virtual environment and install the required packages.

```bash
conda create -n <env_name> python=3.9
conda activate <env_name>
conda install nvidia/label/cuda-12.1.0::cuda
pip install -r requirements.txt
```