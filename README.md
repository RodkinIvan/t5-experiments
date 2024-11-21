# Adaptive Computation Time Investigation

This project explores the integration of [Adaptive Computation Time (ACT)](https://arxiv.org/pdf/1603.08983) into various sequential models, including LSTM, Transformers, [Universal Transformers (UT)](https://arxiv.org/pdf/1807.03819), [Mamba](https://arxiv.org/pdf/1807.03819), and [Associative Recurrent Memory Transformer (ARMT)](https://arxiv.org/pdf/2407.04841). The goal is to dynamically adjust computational steps based on input complexity, enhancing efficiency and performance on diverse tasks.


## Datasets

We evaluate the models on the following datasets:

| Dataset | Description | Train size | Val. size | Test size |
| --- | --- | --- | --- | --- |
| [1D Cellular Automata](https://huggingface.co/datasets/irodkin/1dCA_r2s20T20) | Predicts the next state of a 1D cellular automaton based on its current state | 950,000 | 50,000 | 100,000 |
| [Binary Copy](https://huggingface.co/datasets/steeldream/binary) | Copies the input sequence to the output | 800,000 | 100,000 | 100,000 |
| [Binary Reverse](https://huggingface.co/datasets/steeldream/binary) | Reverses the input sequence | 800,000 | 100,000 | 100,000 |
| [Binary Addition](https://huggingface.co/datasets/steeldream/addition_binary) | Adds two binary numbers | 800,000 | 100,000 | 100,000 |

We preprocess the Binary Copy, Reverse, and Addition datasets using the `collate_fn` function to define their respective tasks. For these datasets, we experimented with two approaches: (1) sampling input lengths to train on strings of varying sizes for better generalization, and (2) using fixed input lengths without sampling. For the Cellular Automata dataset, we evaluated multiple scenarios where the shift (number of look ahead steps) was set to 1, 2, 3, and 4.


# Models

We evaluated the following models on all datasets:

1. LSTM
2. Transformer
3. Mamba
4. ARMT

Each model was evaluated both with ACT applied to each layer (LACT) and to the entire model (MACT). For the Copy and Reverse tasks, all models used a single layer, while for the Addition and Cellular Automata tasks, four layers were used. [GPT-Neox](https://huggingface.co/docs/transformers/model_doc/gpt_neox) was selected as the Transformer model and as the backbone for ARMT.

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

## Training

To run a training script, navigate to the `scripts` directory, select the folder corresponding to the desired dataset/task, and execute the script for the specific model.

For example, to train the Transformer model with one layer using LACT and without sampling input lengths on the Binary Copy task:

```bash
cd scripts
cd copy_binary
bash finetune_gpt_neox_1l_act_no_sample.sh
```