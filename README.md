# Movie-review-sentiment-analysis
This project aims to perform sentiment analysis on movie reviews using the Naive Bayes algorithm. It classifies movie reviews as positive or negative based on the given dataset. The algorithm deals with the challenges of class imbalance, text preprocessing, and feature extraction to achieve accurate predictions.
## Dataset

The dataset consists of movie reviews, both positive and negative. The dataset was preprocessed to balance the classes, remove links, punctuations, stopwords, and perform stemming on the words. The output categories were also converted to a numerical format.

## Preprocessing Steps

1. Upsampling the minority class
2. Removing links, punctuations, and stopwords
3. Lowercasing the text
4. Stemming

## Functions

1. `find_occurrence`: Finds the total occurrence of a word given the label, word, and frequency dictionary.
2. `review_counter`: Counts the occurrence of words and calculates probabilities based on the training data.
3. `naive_bayes_train`: Trains the Naive Bayes model, calculating log likelihood and log prior values.
4. `naive_bayes_predict`: Predicts whether a given review is negative or positive.
5. `naive_bayes_test`: Tests the Naive Bayes model on a test set and returns the total number of correctly classified reviews.

## Model Evaluation

1. Split the data into training and test sets using random selection with a specified seed.
2. Calculate model parameters using the training set.
3. Print the confusion matrix for the training and test sets.
4. Examine False Positive and False Negative cases and provide reasoning for misclassifications.

## Usage of text_classifier.py

1. Launch the application on the command prompt with the following command: `python text_classifier.py`
2. The module will load the model parameters from a local file and be ready to take user input.
3. Enter a movie review and the program will preprocess the input, tokenize it, and predict the sentiment class.
4. The output will display the probabilities for each input token along with the final classification decision.
5. To quit the program, enter "X".



