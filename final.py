import nltk
from collections import defaultdict
import numpy as np


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GRU, Embedding, Concatenate, LSTM, Input, Bidirectional, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import load_model
import keras.models

import random as rn
import tensorflow as tf
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)


fp=open("glove.6B/glove.6B.300d.txt","r")
glove_emb={}
for line in fp:
	temp=line.split(" ")
	glove_emb[temp[0]]=np.asarray([float(i) for i in temp[1:]])


sentences=[]
parts_of_speech=defaultdict(int)
def neural_extractor1(data,categories,aspects,text_to_predict):
	
	for i in range(0,len(data)):
		flag=0
		temp_sent=[]
		text=data[i]
		
		words=nltk.word_tokenize(text)
		pos=[]
		for word in nltk.pos_tag(words):
			parts_of_speech[word[1]]=1
			pos.append(word[1])

		tags=['O' for ff in range(0,len(words))]
		for aspect in aspects[i]:
			asp_words=nltk.word_tokenize(aspect.lower())

			j=0;k=0;
			# flag=0
			while(k<len(asp_words)):
				while(j<len(words)):
					if(asp_words[k]==words[j] and tags[j]=='O'):
						
						if(k==0):
							tags[j]='B'
						else:
							tags[j]='I'
						# if(flag==0):
						# 	tags[j]='B'
						# 	flag=1
						# else:
						# 	tags[j]='I'
						k+=1
						if(k>=len(asp_words)):
							break
					j+=1
				k+=1
		
		for ii in range(0,len(words)):
			temp_sent.append((words[ii],pos[ii],tags[ii]))
		sentences.append(temp_sent)
	print(len(sentences))

	for i in range(0,len(data)):
		tokens = nltk.word_tokenize(data[i])
		string=' '.join(tokens)
		data[i]=string
	#data.append(' '.join(words_to_predict))
	#lll=len(data)-1
	data.append("ENDPAD")
	tokenizer=Tokenizer()
	tokenizer.fit_on_texts(data)
	sequences=tokenizer.texts_to_sequences(data)
	word_index=tokenizer.word_index
	
	
	
	X=pad_sequences(sequences[:-1],maxlen=50,padding="post", value=word_index["endpad"])
	
	validation_size=int(0.2*X.shape[0])
	#print(X_to_predict)

	n_words=len(word_index)

	tag_list=['B','I','O','P']
	n_tags=len(tag_list)

	embedding_matrix = np.zeros((n_words, 300))

	for word,i in word_index.items():
		if(i>=len(word_index)):
			continue
		if word in glove_emb:
			embedding_matrix[i]=glove_emb[word]


	max_len=50
	tag2idx={t:i for i,t in enumerate(tag_list)}
	idx2word={t:i for i,t in word_index.items()}
	pos2idx={t:i for i,t in enumerate(parts_of_speech.keys())}

	y = [[tag2idx[w[2]] for w in s] for s in sentences]
	y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["P"])
	y = [to_categorical(i, num_classes=n_tags) for i in y]

	pos=[[pos2idx[w[1]] for w in s] for s in sentences]
	pos1=pad_sequences(maxlen=max_len, sequences=pos, padding="post", value=len(parts_of_speech.keys())+1)

	pos=np.asarray([np.reshape(i,(max_len,1)) for i in pos1])

	
	# indices=np.arange(X.shape[0])
	# np.random.shuffle(indices)
	# X=X[indices]
	# y=y[indices]
	#validation_size=int(0.2*X.shape[0])
	
	X_tr=X[:-validation_size]
	tr_pos=pos[:-validation_size]
	y_tr=y[:-validation_size]
	X_te=X[-validation_size:]
	te_pos=pos[-validation_size:]
	y_te=y[-validation_size:]
	X_to_predict=X[-validation_size:]
	
	pos_to_predict=pos[-validation_size:]
	# X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)

	vocab_size=len(word_index)
	e=Input(shape=(max_len,))
	emb=Embedding(vocab_size,300,weights=[embedding_matrix],input_length=50,mask_zero=True,trainable=False)(e)
	ad_pos=Input(shape=(max_len,1))
	co_tm=Concatenate()([emb]+[ad_pos])
	bi_gru=Bidirectional(GRU(50,return_sequences=True))(emb)
	out=Dense(25,activation='relu')(bi_gru)
	# out=Dropout(0.1)(out)
	out=TimeDistributed(Dense(n_tags,activation='softmax'))(out)
	model = Model(inputs=[e,ad_pos], outputs=[out])
	model.compile(loss='categorical_crossentropy',optimizer="rmsprop",metrics=['accuracy'])

	model.fit([X,pos], np.array(y), batch_size=25, epochs=20, validation_data=([X_te,te_pos],np.array(y_te)), verbose=1)
	pos_tp=np.asarray([np.asarray([1 for i in range(0,50)]).reshape(50,1)])
	#model=load_model('aspect_extractor.h5')
	#with open('aspect.json') as ff:
	#	model_json=ff.read()
	#	model=keras.models.model_from_json(model_json)
	#model.compile(loss='categorical_crossentropy',optimizer="rmsprop",metrics=['accuracy'])
	#model.load_weights('aspect_weights.h5')
	#model.fit([X], np.array(y), batch_size=25, epochs=15, validation_data=([X_te],np.array(y_te)), verbose=0)
	#print(X_to_predict,X_to_predict.shape)
	p1=model.predict([X_to_predict,pos_to_predict])
	#p1=model.predict([X_to_predict])
	#print(p1)
	pred_aspects=[]
	for i in range(0,len(p1)):
		p=np.argmax(p1[i],axis=-1)
		temp1=[]
		flag=0
		string1=""
		for j in range(0,len(p)):
			#print(idx2word[X_to_predict[i][j]],tag_list[p[j]])
			if(idx2word[X_to_predict[i][j]]=="endpad"):
				break
			if(tag_list[p[j]]=='B'):
				string1+=idx2word[X_to_predict[i][j]]+" "
				if(flag==0):
					flag=1
			elif(tag_list[p[j]]=='I'):
				string1+=idx2word[X_to_predict[i][j]]+" "
			elif(tag_list[p[j]]=='O'):
				if(string1!=""):
					temp1.append(string1)
				string1=""
				flag=0
		pred_aspects.append(temp1)

	#print(pred_aspects)
	return pred_aspects
		
	# print(aspects[:-validation_size][69])

	# for i in range(0,20):
	# 	print(aspects[i],pred_aspects[i])
	
	# p=np.argmax(p,axis=-1)
	# true_p=np.argmax(y_tr[69],axis=-1)

	# for i in range(0,len(p)):
	# 	print(true_p[i],p[i])

	#for w, pred in zip(X_to_predict[0], p1[0]):
	#	print(idx2word[w], tag_list[pred])