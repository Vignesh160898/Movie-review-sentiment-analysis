# -*- coding: utf-8 -*-
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
import pandas as pd
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
warnings.filterwarnings("ignore")

loglikelihood = pd.read_csv("loglikelihood_values.csv")
logprior = 0

def clean_review(review):
    review = review.lower() #Converting the review to lowercase
    review = re.sub(r'[^\w\s]', '', review) #Removing the punctuations in the reviews using regex
    review = re.sub(r'https?://\S+|www\.\S+', '', review) #Removing the links in the review using regex
    review_cleaned = word_tokenize(review) #This function is to tokenize the review
    review_cleaned = [r for r in review_cleaned if r not in stopwords.words("english")] #Removing the stop words from the list of tokens
    wn = nltk.WordNetLemmatizer()
    review_cleaned = [wn.lemmatize(x) for x in review_cleaned] #This function is to lemmatize the review
    review_cleaned = ' '.join(word for word in review_cleaned)


    return review_cleaned


def naive_bayes_predict(review, logprior, loglikelihood):
    word_l = clean_review(review).split()
    #Initialize total_prob to 0
    total_prob = 0 
    #Adding the logprior
    total_prob = total_prob + logprior

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in set(loglikelihood.Word):
            # add the log likelihood of that word to the probability
            total_prob = total_prob + loglikelihood[loglikelihood.Word == word].Loglikelihood.item()
            print('Word:',word,' Probability:' , total_prob)
    return total_prob

while True:
    review = input("Enter your review: ")
    if review != 'X':
        p = naive_bayes_predict(review, logprior, loglikelihood)
        print('The cumulative probability of the words in the review', p)
        if p > 0:
            print('This review is a negative review')
        else:
            print('This review is a positive Review')
    else:
        break










