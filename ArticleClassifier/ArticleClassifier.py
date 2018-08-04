
# coding: utf-8

# In[1]:


import os
from os import path
from collections import Counter
import math


# In[2]:


## Reference: https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html

TRAIN_DATA_PATH = 'bbc_train'
TEST_DATA_PATH = 'bbc_test'
DELIMITERS = [',', '.', '!', '?', '/', '&', '-', ':', ';', '@', '"', "'"]
STOP_WORDS = open('stopwords.txt').read().split()


# In[3]:


## GLOBAL VARIABLES

## Constants
# Optionally use, for the purpose of this I
# hardcoded the values, for use with different
# datasets they are calculated as P(cj) = num(cj)/total(C) during training
article_types = []#['business', 'entertainment', 'politics', 'sport', 'tech']

article_type_probabilities ={
  'business': 450/1979,
  'entertainment': 344/1979,
  'politics': 375/1979,
  'sport': 454/1979,
  'tech': 356/1979  
}

## PARAMETERS COULD BE TUNED
MOST_COMMON_TRAIN = 210
MOST_COMMON_TEST = 10
# Laplace smoothing factor Set to 1, change by inspection
ALPHA = 0.001


# In[4]:


## MORE GLOBAL VARIABLES

# Dictionary of word frequencies of eavh article type
# Key = Article Type
# Value = tuple of word lists and their frequencies
# Value is created by the counter() function which
# creates a list of elements and their numbers
article_types_word_frequencies = {}
# total vocabulary of all articles

### Moved locally
vocabulary = []


# In[5]:


def train_data(train_data_path):
    
    global vocabulary
    global article_types_word_frequencies
    
    for article_type in os.listdir(train_data_path):
        #print(article_type)
        # Add article types to list
        article_types.append(article_type)
        article_type_path = path.join(train_data_path, article_type)
        article_type_words = []
        for article in os.listdir(article_type_path):
            #print(article)
            article_path = path.join(article_type_path, article)
            article_text = open(article_path).read()
            words = article_text.split()
            article_type_words.extend(''.join(w for w in word.lower() if w not in DELIMITERS) for word in words)
        article_type_counter = Counter(article_type_words)        
        # Remove null pointers generated from removing delimiters
        article_type_counter.pop('', None)
        # Remove stopping words
        for stop_word in STOP_WORDS: 
            article_type_counter.pop(stop_word, None)   
            
        article_types_word_frequencies[article_type] = article_type_counter.most_common(MOST_COMMON_TRAIN)
        for word, freq in article_type_counter.most_common(MOST_COMMON_TRAIN):
            vocabulary.extend([word])
    # Remove duplicates - Don't care about order
    vocabulary = list(set(vocabulary))
        
        


# In[6]:


train_data(TRAIN_DATA_PATH)


# In[7]:


# Checking for some stuff
print(len(vocabulary))
for word in vocabulary:
    print(word)


# In[8]:


## Save values of (n) of each article type
n_article_type = {}
for article_type in article_types:
    n = 0
    for word_freq_pair in article_types_word_frequencies[article_type]:
        n += word_freq_pair[1]
    n_article_type[article_type] = n


# In[9]:


# Also Checking
for article_type in article_types:
    print('>>', article_type, n_article_type[article_type]) 


# In[10]:


# Still Checking
for article_type in article_types:
    print('>>', article_type, n_article_type[article_type])
    for word_freq_pair in article_types_word_frequencies[article_type]:        
        print(word_freq_pair[0], word_freq_pair[1])


# In[11]:


def get_word_frequency_given_article_type(word, article_type):
    # Number of occurrences of specified word
    nk = 0
    # Number of words in specified article type
    # n = 0
    #n = n_article_type[article_type]
    for word_freq_pair in article_types_word_frequencies[article_type]:
        if word == word_freq_pair[0]:
            nk = word_freq_pair[1]     
        # n += word_freq_pair[1]    
    # probability = (nk + ALPHA) / (n + ALPHA*len(vocabulary)) 
    # Should maybe save probabilites ??
    return nk+ALPHA


# In[12]:


## Save probabilities of each article type words
# A dictionary of dictionaries
# Key 1: article type
# key 2: word
# Value = probability of word given article type

word_frequencies = {}
for article_type in article_types:
    word_frequencies[article_type] = {}
    for word in vocabulary:
        word_frequencies[article_type][word] = get_word_frequency_given_article_type(word, article_type)


# In[13]:


def predict_test_article(article_path):
    
    #print(article_path)
    global article_type_probabilities
    global article_types
    
    test_article_words = []
    # May need this during analysis
    #test_article_type_probabilities = []
    max_prob = -1000000
    likeliest_article_type = ''
    
    article_text = open(article_path).read()
    words = article_text.split()
    test_article_words.extend(''.join(w for w in word.lower() if w not in DELIMITERS) for word in words)
    article_counter = Counter(test_article_words)
    article_counter.pop('', None)
    for stop_word in STOP_WORDS: 
        article_counter.pop(stop_word, None)
            
    article_words = article_counter.most_common(MOST_COMMON_TEST)
    
    for article_type in article_types:
        # Will sum log(prob) to avoid small probabilities
        article_type_probability = article_type_probabilities[article_type]
        for word in test_article_words:
            word_prob = math.log(word_frequencies[article_type].get(word, ALPHA)) - math.log(n_article_type[article_type] + ALPHA*len(vocabulary))
            article_type_probability += word_prob     
        #print('log: ', article_type, article_type_probability)
        #article_type_probability = math.exp(article_type_probability)
        #print('exp: ', article_type, article_type_probability)
        if(article_type_probability > max_prob):
            max_prob = article_type_probability
            likeliest_article_type = article_type
            
    return  likeliest_article_type
    


# In[14]:


def test_data(test_data_path):
    total_predictions = 0
    total_correct_predictions = 0
    
    for article_type in os.listdir(test_data_path):
        article_type_predictions = 0
        article_type_correct_predictions = 0
        print ('Articles of type:', article_type)
        article_type_path = path.join(test_data_path, article_type)
        for article in os.listdir(article_type_path):
            article_type_predictions += 1
            article_path = path.join(article_type_path, article)
            article_prediction = predict_test_article(article_path)
            print (article, 'Prediction:', article_prediction)
            if article_prediction == article_type:
                article_type_correct_predictions += 1
        print ('Number of predictions: ', article_type_predictions)
        print ('Correct predictions: ', article_type_correct_predictions)
        print ('Accuracy: ', float(article_type_correct_predictions)/article_type_predictions)
        total_predictions += article_type_predictions
        total_correct_predictions += article_type_correct_predictions
                 
    print ('Total Number of predictions: ', total_predictions)
    print ('Correct predictions: ', total_correct_predictions)
    print ('Accuracy: ', float(total_correct_predictions)/total_predictions)            


# In[15]:


test_data(TEST_DATA_PATH)

