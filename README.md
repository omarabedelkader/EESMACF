
# Evaluating the Effectiveness of SBERT and MiniLM on Analogy Classification with FrameNet

# Overview

This project aims to evaluate and compare the performance of two NLP models—SBERT (Sentence-BERT) and MiniLM—for analogy classification using the FrameNet dataset. The goal is to determine the ability of these models to identify valid and invalid analogies by leveraging semantic embeddings and frame relationships. The results demonstrated that a fine-tuned MiniLM model outperformed SBERT in accuracy, achieving 99%.

# Project Structure

- **Introduction**: Understanding the task of analogy classification and its importance in NLP.
- **Problem Definition**: Exploring the use of FrameNet to create and evaluate analogies.
- **Methodology**: Describing transfer learning, data preprocessing, architecture design, and training techniques.
- **Evaluation & Results**: Comparing SBERT and MiniLM performance and analyzing their strengths and limitations.

# Technologies Used

- **SBERT**: Used to generate semantic embeddings of sentences, allowing for comparison of analogy components.
- **MiniLM**: A compact, fine-tunable language model that demonstrated high accuracy in analogy classification tasks.
- **FrameNet Dataset**: A semantic dataset containing annotated frames and sentences, used for training and testing the models.
- **Transfer Learning**: The models were pre-trained and then fine-tuned on FrameNet for improved performance.
- **Python Libraries**: Libraries such as `PyTorch`, `Transformers`, `NLTK`, and `Pandas` were used for model implementation, training, and data preprocessing.

# Installation & Setup

## Prerequisites

- Python 3.7 or higher
- `pip` for package management

## Installing Dependencies

To set up the environment and install necessary dependencies, run:

```bash
pip install -r requirements.txt
```

# Dataset

The FrameNet dataset is used to analyze analogies and train the models. You can download the dataset from [FrameNet](https://framenet.icsi.berkeley.edu/) or use the provided preprocessed version.

# Data Preparation

1. **Data Cleaning**: The dataset undergoes cleaning to remove unwanted characters and symbols.
2. **Tokenization & Lemmatization**: Sentences are tokenized and lemmatized to normalize text.
3. **Analogy Generation**: Analogies are generated in the form A:B::C:D, where valid and invalid pairs are balanced.

# Model Training

Two models were trained and fine-tuned:
- **SBERT**: The pre-trained SBERT model (`distilbert-base-nli-stsb-mean-tokens`) was used to create dense vector embeddings of sentences.
- **MiniLM**: The `microsoft/MiniLM-L12-H384-uncased` model was fine-tuned on the analogy classification task to improve performance.

## Fine-tuning

Hyperparameter optimization was conducted for both models. The best learning rate, batch size, and optimizer were identified for optimal performance:
- SBERT achieved a maximum validation accuracy of 55%.
- MiniLM achieved a maximum validation accuracy of 99%.

# Evaluation

Models were evaluated based on their ability to classify analogies accurately. Training and validation accuracy and loss were recorded, with MiniLM showing a significant improvement over SBERT.

# Results & Conclusions

The fine-tuned MiniLM model demonstrated a high accuracy of 99%, showing a deep understanding of the semantic relationships in FrameNet. This suggests that MiniLM is well-suited for analogy classification and can be applied to various NLP applications like question-answering systems, semantic search, and dialogue agents.

# Future Work

- **Extending Analogy Classification**: Explore additional pre-trained models for further performance gains.
- **Broader Applications**: Apply the models to other semantic tasks in NLP, such as argumentation mining and information extraction.

# References

1. [FrameNet](https://framenet.icsi.berkeley.edu/)
2. Reimers, N., & Gurevych, I. (2019). Sentence-BERT.
3. Wang, W., Xu, R., Qiu, X., & Liu, X. (2020). MiniLM.
