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
from keras.utils import plot_model

data = pd.read_json('./Newdataset.json')
# Keeping only the neccessary columns
data = data[['text','sentiment']]

print(data)
pos = 0
neg = 0
for dados in data["sentiment"]:
	if dados == "Positivo":
		pos +=1
	elif dados == "Negativo":
		neg +=1

print("Positivo:"+str(pos))
print("Negativo:"+str(neg))


data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')
print(data[ data['sentiment'] == 'Positivo'].size)
print(data[ data['sentiment'] == 'Negativo'].size)    
max_fatures = 2000
embed_dim = 128
lstm_out = 196
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

def Mymodel():
	model = Sequential()
	model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
	model.add(SpatialDropout1D(0.4))
	model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(output_dim=2,activation='softmax'))
	return model


model = Mymodel()
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

def Train(model,data,X):
	Y = pd.get_dummies(data['sentiment']).values
	X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
	print(X_train.shape,Y_train.shape)
	print(X_test.shape,Y_test.shape)

	batch_size = 32
	model.fit(X_train, Y_train,validation_data=(X_test,Y_test), epochs = 7, batch_size=batch_size, verbose = 2)
	model.save_weights("./LSTM_trained.h5")
	validation_size = 1500

	X_validate = X_test[-validation_size:]
	Y_validate = Y_test[-validation_size:]
	X_test = X_test[:-validation_size]
	Y_test = Y_test[:-validation_size]
	score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
	print("score: %.2f" % (score))
	print("acc: %.2f" % (acc))
	pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
	for x in range(len(X_validate)):
	    
	    result = model.predict(X_validate[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
	   
	    if np.argmax(result) == np.argmax(Y_validate[x]):
	        if np.argmax(Y_validate[x]) == 0:
	            neg_correct += 1
	        else:
	            pos_correct += 1
	       
	    if np.argmax(Y_validate[x]) == 0:
	        neg_cnt += 1
	    else:
	        pos_cnt += 1
	print("pos_acc", pos_correct/pos_cnt*100, "%")
	print("neg_acc", neg_correct/neg_cnt*100, "%")
Train(model,data,X)

