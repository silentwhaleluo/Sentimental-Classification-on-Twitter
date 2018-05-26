
# coding: utf-8


#import data
path='SOT/'


import time
import re
from functools import partial
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
import csv
import string

start_time=time.time()
#clean data
def CleanData(text):
    #remove unicode
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)       
    text = re.sub(r'[^\x00-\x7f]',r'',text)
    #replace URL
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',text)
    text = re.sub(r'#([^\s]+)', r'\1', text)  
    #replace at urs
    text = re.sub('@[^\s]+','atUser',text)
    #Removes hastag in front of a word
    text = re.sub(r'#([^\s]+)', r'\1', text)
    #Removes integers 
    text = ''.join([i for i in text if not i.isdigit()])
    #Replaces repetitions of exlamation marks 
    text = re.sub(r"(\!)\1+", ' multiExclamation ', text)
    #Replaces repetitions of question marks """
    text = re.sub(r"(\?)\1+", ' multiQuestion ', text)
    #Replaces repetitions of stop marks """
    text = re.sub(r"(\.)\1+", ' multiStop ', text)
    #Removes emoticons from text """
    text = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', text)
    return text


def GetFilteredStem(text):
    #get tokens
    lowers = text.lower()
    #remove the punctuation using the character deletion step of translate
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    no_punctuation = lowers.translate(remove_punctuation_map)
    token=nltk.word_tokenize(no_punctuation)
    #get stems
    #ps = PorterStemmer()
    #stems=[ps.stem(word) for word in token]
    #filter the stopwords
    stop_words = set(stopwords.words('english')+['rt', 'atuser','url','happy','sad'])
    filtered_words = [word for word in token if not word in stop_words]
    return filtered_words

from textblob import TextBlob
def GetPolarity(text):
    text = TextBlob(text)
    if text.sentiment.polarity > 0:
        pol=1
    elif text.sentiment.polarity == 0:
        pol=0
    else:
        pol=-1
    return pol

def CreateDict(key,values):
    document=list(key)
    idx=list(values)
    dict_words=dict(zip(document,idx))
    return dict_words

def GetDictValue(token,diction):
    for i in range(len(token)):
        for j in range(len(token[i])):
            token[i][j]=diction.get(token[i][j])
    return token

def CountWeights(a,b):
    a=set(a)&set(b)
    return int(len(a))

def vectorize_sequences(sequences,dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results



#preprocess the happy dataset:remove duplicates, filter with polarity
'''
happy=pd.read_csv(path+"happy_ori.csv")

#remove duplicates
happy = happy.drop_duplicates() 
print(len(happy))
tweets=happy['tweet.text']

Cleaned_data=[CleanData(text) for text in tweets]

#filter with polarity
#using polarity to do the sentiment analysis for each tweet
polarity=[GetPolarity(text) for text in Cleaned_data]
polarity
happy['Polarity']=polarity
happy=happy[happy.Polarity==1]
#happy=happy.loc[:1000]
print(len(happy))
happy.to_csv(path+'filtered_happy.csv')
print('filtered_happy saved to',path)
del happy
'''

#preprocess the sad dataset:remove duplicates, filter with polarity
'''
sad=pd.read_csv(path+"sad_ori.csv")

#remove duplicates
sad = sad.drop_duplicates() 
print(len(sad))
tweets=sad['tweet.text']

Cleaned_data=[CleanData(text) for text in tweets]

#filter with polarity
#using polarity to do the sentiment analysis for each tweet
polarity=[GetPolarity(text) for text in Cleaned_data]
polarity
sad['Polarity']=polarity
sad=sad[sad.Polarity==-1]
#sad=sad.loc[:1000]
print(len(sad))
sad.to_csv(path+'filtered_sad.csv')
print('filtered_sad saved to',path)

del sad
'''


#preprocess the tweets: remove useless strings, tokenization
happy=pd.read_csv(path+"filtered_happy.csv")
happy=happy.loc[:999]
tweets=happy['tweet.text']
tweets.shape
print(tweets[0])

Cleaned_happy=[CleanData(text) for text in tweets]
print(Cleaned_happy[0])
Filtered_happy=[GetFilteredStem(text) for text in Cleaned_happy]
print(Filtered_happy[0])


sad=pd.read_csv(path+"filtered_sad.csv")
sad=sad.loc[:999]
print(len(sad))
tweets=sad['tweet.text']

Cleaned_sad=[CleanData(text) for text in tweets]
Filtered_sad=[GetFilteredStem(text) for text in Cleaned_sad]
Filtered_sad[0]
print('Text processed')

#combine the have and sad to create the total dictionary

#combe happy and sad tokens
Filtered_data=Filtered_sad+Filtered_happy
from itertools import chain
tweets_tot = list(chain(*Filtered_data))
print('Number of total token',len(tweets_tot))
#create the total dictionary
freqdist = nltk.FreqDist(tweets_tot)
freqdist.plot(30, cumulative=True)
print(len(freqdist))
tweets_freq=sorted(freqdist.items(), key=lambda x:x[1], reverse=True)
dimension=len(freqdist.keys())
print('Number of distinct words',dimension)

#Create the dictionary of bag of words
word_dict=  CreateDict(np.array(tweets_freq)[:,0],list(range(len(np.array(tweets_freq)[:,0]))))
print(len(word_dict))
print('dictionary created')

#repace the tokens to numbers according to the dictionary with descending frequence order
num_token_happy=GetDictValue(Filtered_happy,word_dict)
num_token_sad=GetDictValue(Filtered_sad,word_dict)
num_token_happy[0]

#change the tweets to vector--one-hot
x_happy_train = vectorize_sequences(num_token_happy,len(word_dict))
print(x_happy_train.shape)
print(x_happy_train[1])

x_sad_train = vectorize_sequences(num_token_sad,len(word_dict))
print(x_sad_train.shape)
x_sad_train[1]

#combe the happy and sad to one dataset
happy_train=np.insert(x_happy_train, 0, values=1, axis=1)
sad_train=np.insert(x_sad_train, 0, values=0, axis=1)

#shuffle the dataset
from sklearn.utils import shuffle
happy_train_shuffle = shuffle(happy_train).astype(int)
sad_train_shuffle = shuffle(sad_train).astype(int)
#Here I just use 500 tweets for each dataset because of SAS University limitation. You can use all dataset.
train=np.concatenate((happy_train_shuffle[:500],sad_train_shuffle[:500]), axis=0)
train_shuffle = shuffle(train)

#create the name of each factor
col_name=['Sentiment']
for i in range(1,len(train_shuffle[0])):
    col_name.append('X'+str(i))
len(col_name)
from pandas import DataFrame
train_shuffle=DataFrame(train_shuffle,columns=col_name)
train_shuffle.to_csv(path+'train_1000.csv',index_label ='ID',header=True)
print('DR train data saved to', path)