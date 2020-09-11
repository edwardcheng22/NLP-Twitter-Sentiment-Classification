# NLP Twitter Sentiment Classification

## Business Case

Twitter is a popular social media platform used by hundreds of millions of people around the globe. In fact, the current estimate of twitter users amount to approximately 330 million monthly active users and 145 million daily active users on Twitter. 63 percent of all Twitter users worldwide are between the ages of 35 and 65. The CEO of twitter has tasked me with building a model that has the capabilities of analyzing Twitter sentiments about Apple and Google products. The human raters rated the sentiment in over 9,000 Tweets as positive, negative, or neither.


## Objective of the project
*Build a model that can rate the sentiment as positive, negative, or neither of a tweet based on its content.*


#### Technologies Used:
* Pandas for Data Cleaning
* Matplotlib and Seaborn for Data Visualization
* NLTK for Text Preprocessing
* Scikit Learn for Logistic Regression, Random Forests, Naive Bayes classifier 
* Keras for Neural Network Preprocessing

### Process Overview

## Text Preprocessing

* Clean data set, including checking for NA values

* Stopword Removal
   * Removal of Punctuations and/or numbers
   * Removal of Capitlizations
   * Regex
   
* Stemming/Lemmitazation

* Noise Removal

* Tokenization

 ## Exploratory Data Analysis
 
 ### Length of Tweets
 
 ### Length of Tokenized Words based on the product's emotion
 
 ### Top 20 Most Popular Words
 
 ### Top 20 Most Popular Hashtags
 
 ## Feature Engineering
 
 ### Bigrams/N-grams
 
 ### Mutual Information Scores
 
 ### Text Vectorization
 
 ## Modelling
 
 **6 models tested:**
   
   * Multinomial NB Classifier
   * Logistic Regression
   * Random Forests
   * Support Vector Classifier
   * Deep Neural Networks (Base Model)
   * Deep Neural Networks Regularized (with dropout)
   
## Results

### Final Model Selected:

**Next Steps**
  * Trying different data cleaning methods
    * Having more time to play around with combining unused features together
    * Collect more data to train the model
    * Transforming more categorical variables into numerical ones for modelling purposes
