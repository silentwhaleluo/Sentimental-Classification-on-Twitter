# Simple-sentimental-classification-on-Twitter
Using multiple model to do the classification on Twitter's sentiment
This project is to do sentimental classification on Twitters
## Dataset  
- Extract tweets from Twitter API  
- Clean the data
- Calculate the polarity to help correct the lable
- Natural language preprocessing  
Here is a example of NLP   

The orgininal tweets:

> RT @CricketAus: A happy 50th birthday to one of the game's best exponents of the reverse sweep, Andy Flower! https://t.co/Q2giWl8xMn  

Clean the text (remove numbers and symbol such as '@')

> RT atUser A happy th birthday to one of the game's best exponents of the reverse sweep, Andy Flower! url  

filter the text (Replace uppercase letters with lowercase letters, remove punctuation, remove stopwords('happy','sad'), tokenization)
> 'th', 'birthday', 'one', 'games', 'best', 'exponents', 'reverse', 'sweep', 'andy', 'flower'  

Replace the words with numbers
> [60, 1, 9, 282, 39, 3034, 3035, 3036, 3037, 3038]  

one-hot coding
> [ 0.  1.  0. ...,  0.  0.  0.] (1*4583)


## Models and tranings
### Use 7 different models
- Neural Network	
- Support Vactor Machines
- Bernoulli Naive Bayes	
- Linear DiscrIminant
- Logistic Regression
- Perceptron Learning Algorithm	
- K Neighbors Classifier  
### Train the models and adjust Hyperparameter by validation curve

![Accuracy of different models](https://github.com/silentwhaleluo/Simple-sentimental-classification-on-Twitter/blob/master/Accuracy%20of%20different%20models.png?raw=true)
