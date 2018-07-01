import numpy as np
import nltk,random
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, cross_val_score
# from autocorrect import spell
from sklearn.preprocessing import MultiLabelBinarizer

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GRU, Embedding, LSTM, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras import backend as K
from keras.preprocessing.text import Tokenizer



fp=open("glove.6B/glove.6B.300d.txt","r")
glove_emb={}
for line in fp:
	temp=line.split(" ")
	glove_emb[temp[0]]=np.asarray([float(i) for i in temp[1:]])





def classification(data,labels):
	for i in range(0,len(data)):
		tokens = nltk.word_tokenize(data[i])
		string=' '.join(tokens)
		data[i]=string
	y=MultiLabelBinarizer().fit_transform(labels)

	tokenizer=Tokenizer()
	tokenizer.fit_on_texts(data)
	sequences=tokenizer.texts_to_sequences(data)
	word_index=tokenizer.word_index
	print('Found %s unique tokens.' % len(word_index))

	data=pad_sequences(sequences,maxlen=30)
	print('Shape of data tensor:', data.shape)

	indices=np.arange(data.shape[0])
	np.random.shuffle(indices)
	data=data[indices]
	y=y[indices]
	validation_size=int(0.2*data.shape[0])
	
	X_train=data[:-validation_size]
	y_train=y[:-validation_size]
	X_test=data[-validation_size:]
	y_test=y[-validation_size:]

	print(X_train.shape,X_test.shape)

	


	embedding_matrix = np.zeros((len(word_index), 300))

	for word,i in word_index.items():
		if(i>=len(word_index)):
			continue
		if word in glove_emb:
			embedding_matrix[i]=glove_emb[word]

	vocab_size=len(word_index)

	model=Sequential()
	model.add(Embedding(vocab_size,300,weights=[embedding_matrix],
							input_length=30,
							mask_zero=True,
							trainable=False))

	model.add(LSTM(50))

	model.add(Dense(25,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(5,activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])	

	model.fit(X_train,y_train,batch_size=25,epochs=25,verbose=1,validation_split=0.1)
	loss,acc=model.evaluate(X_test,y_test, verbose=0)
	print(loss,acc)

	# pred=model.predict(X_test)
	# for i in range(0,len(pred)):
	# 	print(pred[i],y_test[i])


	return 