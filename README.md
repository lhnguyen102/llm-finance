# llm-finance
This project is solely for educational and learning purposes, focusing on guiding developers on how to build an LLM model from scratch and then fine-tune it for different tasks. To make a great model, we need to know how every part works and how it uses data. Making this project taught me a lot about that.  In this repository, you will find tools and materials to:

- [x] **Build a core LLM**: Train a foundational language model to understand text
- [x] **Sentiment Analysis Adaptation**: Fine-tune the base LLM to interpret and analyze sentiments in texts.
- [x] **Financial QnA Adaptation**: Adjust the LLM to address question and answer tasks specifically tailored for the financial sector.
- [ ] **Performance Benchmarks (WIP)**: Evaluate the model's effectiveness in sentiment analysis and QnA tasks.
- [ ] **User Interface (WIP)**: An interactive platform to test the model on sentiment analysis and QnA tasks.


The code for training based model are mostly from the repository [llama2.c](https://github.com/karpathy/llama2.c) by Andreij Karpathy. His repository stands as a masterclass of educational content in AI development. Highly recommended for all learners. Here are some main modifications
- Restructured the training code
- Integrated the [8bit optimizer library](https://github.com/TimDettmers/bitsandbytes) for faster base model training
- Made updates to deal with padding in input and output (not quite optimal yet)

All models, from base to fine-tuned, are built from scratch, so you can see how each part works.

## Directory Structure
```shell
.
├── alpaca_finance      # Preprocess Alpaca's QnA dataset
├── config.py           # Configuration for model and network
├── finetune_dataset.py # Dataset and dataloader for the fine-tuning task
├── finetune_model.py   # Fine-tuning model
├── Makefile            # Compiling instruction for C code
├── model.py            # Base model
├── run.c               # C code for forward pass
├── sentiment_finance   # Preprocess the news data for sentiment analyis
├── tinystories.py      # Tiny Stories to train base model
├── tokenizer.bin       # Vocabulary encoder-decoder in binary (C code)
├── tokenizer.model     # Vocabulary encoder-decoder (python code)
├── tokenizer.py        # Script for encodeing and decoding text and token IDs
├── train_base_model.py # Training runner for the base model
└── train_ft_model.py   # Training runner for the fine-tuning model
```

## Installation
### Create Miniconda Environment
We recommend installing miniconda for managing Python environment, yet this repo works well with other alternatives e.g., `venv`.
1. Install miniconda by following these [instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#system-requirements) 
2. Create a conda environment
    ```
    conda create --name your_env_name python=3.10
    ```
3. Activate conda environment
    ```
    conda activate your_env_name
    ```
### Install all required libraries
```shell
pip install -r requirements.txt
```

## Download Dataset
### TinyStories
Run the command below. Note: This script originates from the llama2.c repository but has been slightly modified.
```shell
python tinystories.py download
```

### FinGPT for Sentiment Analysis
Run the following command to preporcess the raw data in the base model input format
```shell
python -m sentiment_finance.make_dataset
```

### Alpaca for Question and Answer
- Download the data `Cleaned_date.json` from [Hugging Face ](https://huggingface.co/datasets/gbharti/finance-alpaca/tree/main).
- Save it the folder `alpaca_finance/data`.
- Then, run the following command to preporcess the raw data in the base model input format.
    ```shell
    python -m alpaca_finance.make_dataset
    ```

## Model Training

### Base Model
Run the follow command to train the base model
```shell
python train_base_model.py training
```

Run the follow command to test the base model
```shell
python train_base_model.py test
```

In order to test the base model in C, you need first to compule the C code by running the following command
```shell
gcc -o run run.c -lm
```
then 
```shell
./run out/model.bin
```
NOTE: you can also use the Makefile to compile the C code if preferred.

### Fine-tuning Model
For fine-tuning, we use HuggingFace's LoRA approach to extract layers but have implemented our own custom optimizer for our custom model. In the future, we plan to implement our minimal version for LoRA approach. Run the following command for fine-tuning model for sentiment analysis
```shell
python train_ft_model.py news
```

Run the following command for fine-tuning model for questions and anwsers
```shell
python train_ft_model.py alpaca
```

## Model Testing
Coming soon...
## Acknowledgement
### Dataset

- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories): Provided the data for training the base model
- [FINGPT](https://github.com/AI4Finance-Foundation/FinGPT): Provided the data for the sentiment analysis
- [Gaurang Bharti](https://huggingface.co/datasets/gbharti/finance-alpaca): Put together data from [Standford's Alpaca](https://github.com/tatsu-lab/stanford_alpaca) and [FiQA](https://sites.google.com/view/fiqa/) for question and answer finetuning

