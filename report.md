# News category classification report
train.py script
This simple model reads news headlines and short descriptions, preprocesses them, trains a SVM classifier to predict categories based on the text.

## eval.py script
It loads a trained model, evaluates its performance on a separate dataset, and provides metrics such as accuracy, confusion matrix, and classification report to assess the model's effectiveness.

## classify.py script
This script takes test data, classifies it using a pre-trained model, prints the distribution of predicted categories, and saves the classified data to a jsonl file.

## .ipynb files
- In the .ipynb files, I explored various techniques for feature extraction and model training:
- **Feature Extraction**: Techniques like TF-IDF (Term Frequency-Inverse Document Frequency) and word embeddings (Word2Vec) were implemented for feature extraction from the text data.
- **Model Selection**: Different models were applied for classification, including Naive Bayes, Support Vector Machines (SVM), Decision Trees, Random Forests, and deep learning models.
- **Optimization Techniques**: Optimization techniques such as grid search and random search were employed to fine-tune hyperparameters and improve model performance.

Two different approaches were implemented in Google Colab to observe the data:

1. **Deep Learning Model with Different Classifiers**: In one approach, a deep learning model was implemented with various classifiers for news category classification.

2. **Simple Model with Different Classifiers**: In the other approach, a simpler model architecture was used, and multiple classifiers were applied to classify news categories.

## Possible improvements
Applying transfer learning, such as leveraging pre-trained language models like BERT and GPT, can significantly enhance performance. Fine-tuning these models on the dataset can lead to significant performance gains.
