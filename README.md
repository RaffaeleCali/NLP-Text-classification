# NLP Project - Text Classification

## Author
Raffaele Calì 

## Description
This project explores various strategies for text classification, categorizing texts into five categories: business, sports, politics, technology, science, and 'other'. The project combines deep learning techniques and classical machine learning to achieve the best possible results.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Techniques Used and Results](#techniques-used-and-results)
4. [Conclusions and Future Developments](#conclusions-and-future-developments)
5. [Demo](#demo)

## Introduction
The project focuses on text classification, defined as the process of categorizing text into organized groups. The goal is to classify texts into five categories using deep learning techniques, classical machine learning, and a combination of both.
For more information open `relazione.pdf`.
## Dataset
The final dataset is a combination of three datasets: BBC News, AG News, and 20Newsgroup. These have been mapped into five predefined categories to ensure consistency. The dataset contains approximately 130,000 records with a balanced distribution across categories.

### Dataset Structure
- **BBC News**: Formal journalistic articles.
- **AG News**: Brief and concise news summaries.
- **20Newsgroup**: Forum messages with an informal style.

The dataset is divided into two parts:
- `dataset_k_neigh.csv`
- `dataset_Longformer.csv`

A secondary dataset, `generated_pairs.csv`, was created for the binary classification task.

## Techniques Used and Results

### 1. Baseline: CLS Token Extraction from Longformer and Classification with KNN (CLS.ipynb)
- **Procedure**: Text preprocessing, extraction of CLS embeddings with Longformer, classification with KNN.
- **Results**:
  - Accuracy: 0.891243
  - Precision: 0.898594
  - Recall: 0.823507
  - F1: 0.850655

### 2. LDA and KNN (CLS_LDA.ipynb)
- **Procedure**: Topic configuration with LDA, text analysis, classification with KNN.
- **Results**:
  - Accuracy: 0.730223
  - Precision: 0.659870
  - Recall: 0.620796
  - F1: 0.625178

### 3. Fine-tuning Longformer + KNN (binary_def.ipynb, FT_CLS_LDA.ipynb, and FT_CLS.ipynb)
- **Procedure**: Fine-tuning Longformer on a binary classification task, extracting CLS embeddings, classification with KNN.
- **Results**:
  - CLS: 
    - Accuracy: 0.891243
    - Precision: 0.898594
    - Recall: 0.823507
    - F1: 0.850655
  - CLS+LDA:
    - Accuracy: 0.882041
    - Precision: 0.879741
    - Recall: 0.820107
    - F1: 0.843028

### 4. Fine-tuning Longformer with Multiclass Classification (deflong-multiclass.ipynb)
- **Procedure**: Fine-tuning Longformer for multiclass classification.
- **Results**:
  - Accuracy: 0.8855
  - Precision: 0.8733
  - Recall: 0.8545
  - F1: 0.8624

## Conclusions and Future Developments
Fine-tuning Longformer for a multiclass classification task has shown the best performance. For future developments, it is recommended to further explore methods for handling global attention in transformer models.

## Demo
To try the demo:
1. Run the `demo.py` file.
2. Enter the text in the textarea.
3. Click on "Classify text" to see the results.
---
