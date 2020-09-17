# NLP Twitter Sentiment Classification

![Twitter](https://github.com/edwardcheng22/NLP-Twitter-Sentiment-Classification/blob/master/images/twitter_image.jpeg)

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

![Process](https://github.com/edwardcheng22/NLP-Twitter-Sentiment-Classification/blob/master/images/process.PNG)

## Text Preprocessing

* Clean data set, including checking for NA values

* Stopword Removal: NLTK has a built in method that captures all of the stopwords in the english dictionary. The purpose of eliminating stopwords is for less frequently occurring, but potentially very informative words to be given more importance in the modelling stage. Let's take this sentence as an example: 'Today is going to be a beautiful day.' Stopwords would include words such as 'is' and 'a'. If your task was to determine the mood/emotion behind this text sentence, you would probably say that the word 'beautiful' provides the most positive meaning in the sentence. If you do not remove stopwords, then words such as 'is' and 'a' will essentially be given the same importance as the word 'beautiful', which is not something that you want when performing text classification analysis.
   * Removal of Punctuations and/or numbers: A continuation of stopword elimination.
   * Removal of Capitlizations: All words should be lowercased for consistency purposes.
   * Regex: This can be a very handy tool if you want to customize your stopword list and add more words/characters/symbols that you want to eliminate from your text data.
   
* Stemming/Lemmitazation
   * Stemming and Lemmatization are both similar in that they are used to return a condensed version of a corpus of words. For example, these techniques would reduce the word 'running' to 'run'. The difference between Stemming and Lemmatization is that stemming does not account for the context of the words as much as lemmatization. For example,  the word 'ponies' would be reduced to 'poni'. As you can see, Stemming just eliminates the ending of the word to provide a 'rooted' version of the word. Lemmatizing, however, covers for this downfall of stemming by having built in methods to detect root words in the English dictionary and stems the word correctly. Lemmatizing the word 'ponies' would result in 'pony'. The advantage of stemming is that it produces results much quicker, but lemmatizing provides more accurate results albeit it does take more time to complete, but that's a tradeoff for you to decide.

* Noise Removal: Remove other unnecessary words/characters/symbols

* Tokenization: Tokenization basically refers to splitting up a larger body of text into smaller lines, words or even creating words for a non-English language. It can be useful if you want to explore your data in detail and see what the most popular words are, or which hashtags were the most important, etc. In other words, tokenizing the words essentially allows you to visualize your words better. The following two snippets are my text data that has been cleaned, one of which is not tokenized and the other is.

 ## Cleaned Non-Tokenized Version
 
 ![Cleaned](https://github.com/edwardcheng22/NLP-Twitter-Sentiment-Classification/blob/master/images/cleaned_data.png)

 ## Cleaned Tokenized Version
 
 ![Cleaned_tokens](https://github.com/edwardcheng22/NLP-Twitter-Sentiment-Classification/blob/master/images/tokenized_version.PNG)
 

 ## Exploratory Data Analysis
 
 There are 3 categories for the target variable product's emotion: positive, negative, and neutral. I converted them into integer values, 0 representing positive, 1 for negative, and 2 for neutral for modelling purposes.
 
 ### Length of Tweets based on product's emotion
 
 ![tweet_length](https://github.com/edwardcheng22/NLP-Twitter-Sentiment-Classification/blob/master/images/tweet_length.PNG)
 
 ### Length of Tokenized Words based on the product's emotion
 
 ![token_length](https://github.com/edwardcheng22/NLP-Twitter-Sentiment-Classification/blob/master/images/token_length.PNG)
 
 ### Top 20 Most Popular Words
 
 ![top_20_words](https://github.com/edwardcheng22/NLP-Twitter-Sentiment-Classification/blob/master/images/top_20_popular_words.PNG)
 
 ### Top 20 Most Popular Hashtags
 
 ![hashtags](https://github.com/edwardcheng22/NLP-Twitter-Sentiment-Classification/blob/master/images/top_20_popular_hashtags.PNG)
 
 ## Feature Engineering
 
 In this section, I used the tokenized version of my text data set and explored the top pairs of words and the 3-worded combinations. In the snippet shown above, it only showed the most popular individual words. Knowing individual word frequencies is somewhat informative, but in practice, some of these tokens are actually parts of larger phrases that should be treated as a single unit. This is where bigrams and n-grams come into play, as NLTK has a built in method that can do the job of outputting the most frequent pairs of words.
 
 ### Bigrams
 
 ![bigrams](https://github.com/edwardcheng22/NLP-Twitter-Sentiment-Classification/blob/master/images/bigrams.PNG)
 
 The results aren't too surprising, as you can hardly imagine the top pair of words not being treated together as a single unit, such as 'apple store'. Let's take a look now at the top 3-worded combinations.
 
 ### N-grams (3 words)
 
 ![trigrams](https://github.com/edwardcheng22/NLP-Twitter-Sentiment-Classification/blob/master/images/3-gram-combination.PNG)
 
 Generally, the results are pretty similar to the bigrams. Now, let's take a look at an extension of bigrams/n-grams with the concept of mutual information scores. This method measures the mutual dependence between two words. I was required to set up a frequency filter and a threshold, which I deemed to be 10. This number represented the minimum number of times a bigram combination must occur. Performing this action eliminates the less frequently seen bigrams and thus shines a spotlight on the most frequently occuring ones, which can be very informative.
 
 ### Mutual Information Scores
 
 ![mutual](https://github.com/edwardcheng22/NLP-Twitter-Sentiment-Classification/blob/master/images/mutual_info_score.PNG)
 
 The results shown here are quite surprising frankly, as I noticed that the bigrams are pretty randomly spread out with no clear indication of what the bigrams are talking about. The bigrams also do not seem to really be talking about tech-related content.
 
 ### Text Vectorization
 
 This is the final stage before modelling. I first need to vectorize our data set because machines can only recognize numerical values, not text data. There are several different methods to vectorize the data set, including CountVectorization, TF-IDF, and word2vec. For this scenario, I will elect to use the TF-IDF. Count vectorization counts the number of times a word matches a word in the english dictionary but does not take into account the context behind words. For example, if we were looking at a sentence that was 'Today is a beautiful day', the context behind the sentence would be a happy/positive mood. What words generate this positive mood? You would probably say beautiful. In Count vectorization, the word 'a' would be treated with equal value as the word 'beautiful'. TF-IDF solves this issue by assigning a score to words by normalizing them. Essentially, TD-IDF highlights each word’s relevance in the entire document. After using sklearn's built in TFidfVectorizer method, I converted the vectorized text data into an array for modelling purposes.
 
 ## Modelling
 
 **6 models tested:**
   
   * Multinomial NB Classifier
   * Logistic Regression
   * Random Forest
   * Support Vector Classifier
   * Deep Neural Network (Base Model)
   * Deep Neural Network Regularized (with dropout)
   
## Results

### Final Model Selected: Deep Neural Network Regularized (with dropout)

## Final Accuracy Score: 75.4%

![sgd](https://github.com/edwardcheng22/NLP-Twitter-Sentiment-Classification/blob/master/images/SGD_model.PNG)

![results](https://github.com/edwardcheng22/NLP-Twitter-Sentiment-Classification/blob/master/images/results.PNG)

# Check for Overfitting/Underfitting

![loss](https://github.com/edwardcheng22/NLP-Twitter-Sentiment-Classification/blob/master/images/loss_function.PNG)
![accuracy](https://github.com/edwardcheng22/NLP-Twitter-Sentiment-Classification/blob/master/images/accuracy_function.PNG)

**Next Steps**
  * Trying different data cleaning methods
    * Having more time to play around with combining unused features together
    * Collect more data to train the model
    * Transforming more categorical variables into numerical ones for modelling purposes
