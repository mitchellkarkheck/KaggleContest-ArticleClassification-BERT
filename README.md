# KaggleContest-ArticleClassification-BERT

This repository contains code for a machine learning model that classifies scientific research papers into one of 20 categories. The data used in this project is from a Kaggle in-class competition, where each research paper is labeled with a numerical ID corresponding to different subject areas. For each research paper, the dataset contains an ID, the title, and the abstract.

## Dataset Description

- **ID**: A unique identifier for each research paper.
- **Title**: The title of the research paper.
- **Abstract**: The abstract of the research paper.
- **Labels (0-19)**: Numerical IDs corresponding to 20 different subject areas.

## Dependencies

To install dependencies, run: requirements.txt

Included are :
torch for building and training the neural network models.
transformers for using the SciBERT model.
pandas for handling the dataset.
tqdm for displaying progress bars during training.
scikit-learn for evaluating the model.
matplotlib for plotting training and validation losses.
seaborn for creating a confusion matrix heatmap.

## General Explanation of Code

Data Preprocessing:
The dataset is loaded from CSV files and preprocessed. This involves tokenizing the text and converting them into tensors that can be fed into the SciBERT model. This is done using the BertTokenizer from the transformers library. During this step, the text is tokenized, and attention masks are created to ignore padding tokens. The data is split into training and validation sets and is loaded using DataLoader, which helps in efficient loading of data in batches.

Model:
The main model used is SciBERT, which is a BERT model trained on scientific text. A classifier is built on top of SciBERT by adding a linear layer. This linear layer is used to predict the category labels from the SciBERT embeddings. Google Colab GPU was used for faster training.

Training:
The training process involves two stages: pre-training and main training. The code uses a custom pretraining function called pretrain to initialize the model's weights in a more favorable state for the task. During training, the model is fed input data, and the weights are adjusted to minimize the loss, which measures the difference between the predicted labels and the actual labels. The AdamW optimizer and a learning rate scheduler with warmup are used to adapt the learning rate during training.

The training loss is calculated for each batch and backpropagated through the model to update the weights. The validation loss is also computed at the end of each epoch. Both training and validation losses are saved for later analysis.

Visualization:
A plot is generated to display the training and validation losses as the model trains. This helps in analyzing how well the model is learning and whether it is overfitting or underfitting.

Evaluation:
After training, the model is evaluated on the validation dataset. A classification report is printed, which shows precision, recall, and F1-score for each class. Additionally, a confusion matrix is plotted to visualize the performance of the model across different classes.

Submission:
The model is then used to make predictions on a test dataset, and the predictions are saved to a CSV file. This CSV file was then submitted to a Kaggle competition.





