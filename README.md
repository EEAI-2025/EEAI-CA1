# EEAI-CA1: Multi-Label Email Classification - Chained Multi-Outputs

## Overview
This project is part of the CA assessment for the **Engineering and Evaluating Artificial Intelligence** module at NCI. The objective of this coding implementation is to extend an existing email classification system to support **multi-label classification** using **Chained Multi-Output Classification**.

## Project Goal
The current system only supports **multi-class classification** (one label per email). In this implementation, we modify the architecture to allow **multi-label classification**, meaning an email can have multiple labels across different dependent variables.

We achieve this by implementing **Design Decision 1: Chained Multi-Outputs**, where labels are predicted sequentially, considering the dependencies between them.

## Design Choice: Chained Multi-Outputs
- The system classifies **Type 2** first.
- The predicted output of **Type 2** is used as an additional feature for classifying **Type 3**.
- The predicted outputs of **Type 2 & Type 3** are then used to classify **Type 4**.

This ensures that dependencies between labels are respected, improving classification accuracy for multi-label email categorization.

## Getting Started
1. Clone the repository.
2. Run the main script to preprocess data and train models.
```sh
python main.py
```

