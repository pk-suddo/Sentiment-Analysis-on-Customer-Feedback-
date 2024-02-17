# Restaurant Review Sentiment Analysis

## Overview
This repository contains a Python script for sentiment analysis on restaurant reviews. The project utilizes Natural Language Processing (NLP) 
techniques and various machine learning models to classify reviews into positive or negative sentiments. It preprocesses the text data to remove 
stopwords and applies stemming before vectorizing the reviews. The script then evaluates and compares the performance of multiple classifiers to 
select the best model for prediction.

## Features
* Text preprocessing including cleaning, lowercasing, and stemming.
* Removal of specific stopwords to retain sentiment context.
* Text vectorization using CountVectorizer.
* Evaluation of multiple classifiers: Logistic Regression, Naive Bayes, Support Vector Machine, and Random Forest.
* Selection of the best model based on accuracy.
* Model saving for future sentiment prediction on new reviews.
  
## Prerequisites

To run this script, you will need Python and the following libraries:
* NumPy
* Matplotlib
* Pandas
* NLTK
* Scikit-learn
* Joblib
  
## Installation
* 		Clone this repository to your local machine.
* 		Ensure you have Python installed.
* 		Install the required Python libraries using the following command:bash  Copy code pip install numpy matplotlib pandas nltk scikit-learn joblib   
* 		Download the NLTK stopwords dataset by running the following Python commands:python  Copy code import nltk nltk.download('stopwords')   

## Usage

To perform sentiment analysis on restaurant reviews:
* 		Place your dataset in the same directory as the script, naming it Restaurant_Reviews.csv.
* 		Run the script using Python:
*
* 		bash
* 		 python sentiment_analysis.py
* 		
* 		The script will preprocess the reviews, evaluate classifiers, and save the best model along with the vectorizer to disk. You can then use these saved models to predict the sentiment of new reviews.
* 
## Contributing

Contributions to improve the script or extend its functionality are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.




