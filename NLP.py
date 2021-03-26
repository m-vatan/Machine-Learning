#!/usr/bin/env python
# coding: utf-8

# In[232]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import re
import nltk
import nltk.tokenize
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, Conv1D
from keras.layers.recurrent import LSTM, GRU,SimpleRNN
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud


# In[233]:


df = pd.read_csv('Sentiment140.tenPercent.sample.tweets.tsv', delimiter='\t')


# In[234]:


df['sentiment_label'].value_counts()


# In[235]:


#checking for duplicate and missing values
df.drop_duplicates(subset='tweet_text',inplace=True)
df.isnull().sum()


# In[236]:


df = df.reset_index(drop=True)


# In[237]:


df['sentiment_label'].value_counts()


# In[238]:


df.info()


# In[239]:


df.head()


# In[240]:


import seaborn as sns 

sns.countplot(x='sentiment_label', data=df)


# In[241]:


# get a word count per sentence column

def word_count(sentence):
    return len(sentence.split())

df['word count'] = df['tweet_text'].apply(word_count)
df


# In[242]:


import matplotlib.pyplot as plt
#plot word count distribution for both + and - sentiment
x = df['word count'][df.sentiment_label == 4]
y = df['word count'][df.sentiment_label == 0]

plt.figure(figsize=(12,6))
plt.xlim(0,45)
plt.xlabel('word count')
plt.ylabel('frequency')
g = plt.hist([x,y], alpha=0.5, label=['Positive', 'Negative'])
plt.legend(loc='upper right')


# In[243]:


from collections import Counter

# get most common work in training dataset
all_words = []
for line in list(df['tweet_text']):
    words = line.split()
    for word in words:
        all_words.append(word.lower())
        
Counter(all_words).most_common(10)


# In[244]:


# plot word frequency distribution of firdt few words
plt.figure(figsize=(12,5))
plt.title('Top 25 most common words')
plt.xticks(fontsize=13, rotation=90)
fd = nltk.FreqDist(all_words)
fd.plot(25, cumulative=False)

#log-log
word_counts = sorted(Counter(all_words).values(), reverse=True)
plt.figure(figsize=(12,5))
plt.loglog(word_counts, linestyle='-', linewidth=1.5)
plt.ylabel("Frequency")
plt.xlabel('Word Rank')
plt.title('log-log plot of words frequency')


# In[ ]:





# In[245]:


stops = set(stopwords.words('english'))

def cleanText(text): 
    text = re.sub(r'@[A-Za-z0-9]+',' ', text)     #remove @mention
    text = re.sub(r'(?:\@|https?\://)\S+', ' ', text) #remove URLs
    #text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)   # Remove single characters from the start
    text = re.sub(r'\s+[a-zA-Z]\s+',' ', text)    # Single character removal
    text = re.sub('[^a-zA-Z]', ' ', text)       # Remove punctuations and numbers
    text = re.sub(r'\W', ' ', str(text))         # Remove all the special characters
    text = re.sub(r'\b\w{1,2}\b', ' ', text)    # Remove words with 2 or fewer letters
    text = re.sub(r'\s+',' ', text)              # Removing multiple spaces 
    text = text.lstrip().lower()
    removed_stop =  [word for word in text.split() if word not in stops]
    #removed_stop = ' '.join(removed_stop)
    return removed_stop


# In[246]:


df['tweet_text'] =df['tweet_text'].apply(cleanText)


# In[247]:


df['tweet_text'][:10]


# In[248]:


# get most common work in training dataset
X = df['tweet_text']
all_words = []
for i in range(len(X)):
    words = X[i]
    for word in words:
        all_words.append(word.lower())
        
Counter(all_words).most_common(20)
# plot word frequency distribution of first few words
plt.figure(figsize=(12,5))
plt.title('Top 25 most common words')
plt.xticks(fontsize=13, rotation=90)
fd = nltk.FreqDist(all_words)
fd.plot(25, cumulative=False)

#log-log
word_counts = sorted(Counter(all_words).values(), reverse=True)
plt.figure(figsize=(12,5))
plt.loglog(word_counts, linestyle='-', linewidth=1.5)
plt.ylabel("Frequency")
plt.xlabel('Word Rank')
plt.title('log-log plot of words frequency')


# In[249]:


# split sentences to get individual words

sen = list(df['tweet_text'])
all_words = []
for i in range(len(X)):
    words = X[i]
    for word in words:
        all_words.extend(word.split())
    
# create a word frequency dictionary
wordfreq = Counter(all_words)

# draw a Word Cloud with word frequencies
wordcloud = WordCloud(width=900,
                      height=500,
                      max_words=500,
                      max_font_size=100,
                      relative_scaling=0.5,
                      colormap='Blues',
                      normalize_plurals=True).generate_from_frequencies(wordfreq)

plt.figure(figsize=(17,14))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[250]:


df.drop('word count', axis=1, inplace=True)


# In[251]:


X = df['tweet_text']
X[:10]


# In[252]:


y = df['sentiment_label'].values
y = np.array(list(map(lambda x:1 if x == 4 else 0, y)))
len(y)


# In[253]:


from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open('glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()


# In[254]:


embeddings_dictionary


# In[255]:


embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


# In[256]:


from keras.preprocessing.text import Tokenizer


tokenizer = Tokenizer(num_words=5000)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

maxlen = 100


# In[257]:




#Tokenize
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)

X_tokenize = tokenizer.texts_to_sequences(X)
#X_test_tokenize = tokenizer.texts_to_sequences(X_test)


# In[258]:


X_tokenize[:10]


# In[259]:


from keras.preprocessing.sequence import pad_sequences

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

X_pad = pad_sequences(X_tokenize, padding='post')


# In[260]:


X_pad[0]


# In[261]:


#Split dataset to training and test dataset 
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=.3, 
                                                    random_state=42,
                                                    shuffle=True)


# In[262]:


print('X_train = ', X_train.shape)
print('X_test = ', X_test.shape)
print('y_train = ', y_train.shape)
print('y_test = ', y_test.shape)
X_train


# # LogisticRegression
# 

# In[120]:


from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# In[121]:


#X, y = make_classification(random_state=42, shuffle=True)


# In[122]:


pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=500))


# In[123]:


pipe.fit(X_train, y_train)  # apply scaling on training data


# In[124]:


pipe.score(X_test, y_test)  


# # Decision Tree classifier

# In[125]:


from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier()


# In[126]:


DTC.fit(X_train, y_train)


# In[127]:


from sklearn.metrics import accuracy_score


# In[128]:


accuracy_score(DTC.predict(X_test), y_test)


# # Random Forest Classification
# 

# In[129]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import classification_report 


# In[130]:


RFC = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42) 


# In[131]:


RFC_model = RFC.fit(X_train, y_train)


# In[132]:


y_pred = RFC_model.predict(X_test)


# In[133]:


print(classification_report(y_test, y_pred))


# In[134]:


print(precision_recall_fscore_support(y_test, y_pred))


# # SGD

# In[135]:


from sklearn.linear_model import SGDClassifier


# In[136]:


clf = SGDClassifier(loss='modified_huber', alpha=0.01, penalty='l2', max_iter=1000, learning_rate='optimal')


# In[137]:


clf.fit(X_train, y_train)


# In[138]:


y_pred = clf.predict(X_test)


# In[139]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[140]:



# Confusion Matrix
matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix = \n', matrix)

# Classification Report
print('\nClassification Report')
report = classification_report(y_test, y_pred)
print(report)

# Accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('SGD Classifier Accuracy of the model: {:.2f}% '.format(accuracy*100))


# # CNN
# 

# In[153]:


model = Sequential()

embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)

model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


# In[154]:


print(model.summary())


# In[155]:


history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)


# In[156]:


score = model.evaluate(X_test, y_test, verbose=1)


# In[157]:


print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[158]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()


# # RNN

# In[159]:


model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)
model.add(LSTM(128))
#model.add(SimpleRNN(100))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


# In[160]:


print(model.summary())


# In[161]:


history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)


# In[162]:


score = model.evaluate(X_test, y_test, verbose=1)


# In[163]:


print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[164]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


# In[ ]:





# In[ ]:




