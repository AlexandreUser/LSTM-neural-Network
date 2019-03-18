import numpy as np 
import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from keras.layers import  Convolution1D, Flatten, Dropout
import json

max_fatures = 2000
embed_dim = 128
lstm_out = 196
data = pd.read_json('./Newdataset.json')
# Keeping only the neccessary columns
data = data[['text','sentiment']]
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)


def Mymodel():
	model = Sequential()
	model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
	model.add(SpatialDropout1D(0.4))
	model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(2,activation='softmax'))

	return model
def Load_model(model,model_path):
	model.load_weights(model_path)
	model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
	return model

def text_sentiment(model,text):
	text = [text]
	text = tokenizer.texts_to_sequences(text)
	#padding the tweet to have exactly the same shape as `embedding_2` input
	text = pad_sequences(text, maxlen=58, dtype='int32', value=0)
	print(text)
	sentiment = model.predict(text,batch_size=1,verbose = 2)
	print (sentiment)
	sentiment = sentiment[0]
	if(np.argmax(sentiment) == 1):
	    print("positive")
	    print(sentiment)
	    print(np.argmax(sentiment))
	else:
	    print("Negative")
	    print(sentiment)
	    print(np.argmax(sentiment))
	return np.argmax(sentiment)

model_path = "./LSTM_trained.h5"
model = Mymodel()
model = Load_model(model,model_path)
while True:
	texto = input("Insira seu texto: ")
	text_sentiment(model,texto)
