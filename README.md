# AI_text_detection
 
The dataset is from https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset

This Python script implements a text classification model for AI text detection, leveraging the BERT model. Initially, it tokenizes the text dataset with the BERT tokenizer. Subsequently, it constructs a neural network model named `AITextDetectionModel` based on the architecture of the BERT model. The model is specifically designed for binary classification, featuring two classes: "NOT AI" and "AI TEXT." Following the definition of parameters, the model undergoes training on both the validation and test sets.

In the final phase, the script utilizes a sample text to assess the model. This involves preprocessing the text, predicting the label ("NOT AI" or "AI TEXT"), and displaying both the predicted label and the corresponding confidence scores.
