import nltk
from collections import defaultdict
import numpy as np


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GRU, Embedding, LSTM, Input, Bidirectional, TimeDistributed, merge, Reshape, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


fp=open("glove.6B/glove.6B.300d.txt","r")
glove_emb={}
for line in fp:
	temp=line.split(" ")
	glove_emb[temp[0]]=np.asarray([float(i) for i in temp[1:]])


sentences=[]
kk_sentences=[]
pos_tags=[]
def aspect_sentiment(data,aspects,sentiments,kk,kk_aspects):
	for i in range(0,len(kk)):
		temp_sent=[]
		text=kk[i].lower()
		words=nltk.word_tokenize(text)
		pos_kk=[]
		for word in nltk.pos_tag(words):
			pos_kk.append(word[1])
			
		tags=['O' for ff in range(0,len(words))]
		dist=[]

		for aspect in kk_aspects[i]:
			asp_words=nltk.word_tokenize(aspect)
			
			j=0;k=0;
			flag=0
			while(k<len(asp_words)):
				while(j<len(words)):
					if(asp_words[k]==words[j] and tags[j]=='O'):
						if(flag==0):
							tags[j]='B'
							dist.append(j)
							flag=1
						else:
							tags[j]='I'
						k+=1
						if(k>=len(asp_words)):
							break
					j+=1
				k+=1
		for d in range(0,len(dist)):
			flag=0
			for ii in range(0,len(words)):
				if(d==ii):
					flag=1
				if(tags[ii]=='O'):
					flag=0
				if(tags[ii]=='I' and flag==1):
					distance=0
				else:
					distance=ii-dist[d]
				temp_sent.append((words[ii],distance,pos_kk[ii]))
			kk_sentences.append((temp_sent,kk_aspects[i][d],kk[i]))
	i=0
	print(len(data),len(aspects))
	for i in range(0,len(aspects)):
		temp_sent=[]
		text=data[i]
		words=nltk.word_tokenize(text)
		pos=[]
		for word in nltk.pos_tag(words):
			pos.append(word[1])
			pos_tags.append(word[1])
		tags=['O' for ff in range(0,len(words))]
		dist=[]

		for aspect in aspects[i]:
			
			asp_words=nltk.word_tokenize(aspect)
			
			j=0;k=0;
			flag=0
			while(k<len(asp_words)):
				while(j<len(words)):
					if(asp_words[k]==words[j] and tags[j]=='O'):
						if(flag==0):
							tags[j]='B'
							dist.append(j)
							flag=1
						else:
							tags[j]='I'
						k+=1
						if(k>=len(asp_words)):
							break
					j+=1
				k+=1
		for d in range(0,len(dist)):
			flag=0
			for ii in range(0,len(words)):
				if(d==ii):
					flag=1
				if(tags[ii]=='O'):
					flag=0
				if(tags[ii]=='I' and flag==1):
					distance=0
				else:
					distance=ii-dist[d]
				temp_sent.append((words[ii],distance,pos[ii]))
			sentences.append((temp_sent,sentiments[i][d],aspects[i][d],data[i]))
	

	list_of_pos=list(set(pos_tags))
	pos2idx={t:i for i,t in enumerate(list_of_pos)}

	word_list=[]
	for i in range(0,len(data)):
		tokens = nltk.word_tokenize(data[i])
		word_list.extend(tokens)
		string=' '.join(tokens)
		data[i]=string
	data.append("ENDPAD")
	word_list.append("endpad")
	wordss=list(set(word_list))
	word_index={w:i for i,w in enumerate(wordss)}

	# tokenizer=Tokenizer()
	# tokenizer.fit_on_texts(data)
	# sequences=tokenizer.texts_to_sequences(data)
	# word_index=tokenizer.word_index

	X=[[word_index[w[0]] for w in s[0]] for s in sentences]
	X_to_predict=[[word_index[w[0]] for w in s[0]] for s in kk_sentences]
	dist_to_predict=[[w[1] for w in s[0]] for s in kk_sentences]
	distances=[[w[1] for w in s[0]] for s in sentences]
	pos_l=[[pos2idx[w[2]] for w in s[0]] for s in sentences]
	pos_to_predict=[[pos2idx[w[2]] for w in s[0]] for s in kk_sentences]

	X=pad_sequences(X,maxlen=50,padding="post", value=len(word_index.keys())-1)
	D1=pad_sequences(distances,maxlen=50,padding="post", value=50)
	P1=pad_sequences(pos_l,maxlen=50,padding="post", value=len(list_of_pos))
	
	X_to_predict=pad_sequences(X_to_predict,maxlen=50,padding="post", value=len(word_index.keys())-1)
	D_to_predict1=pad_sequences(dist_to_predict,maxlen=50,padding="post", value=50)
	P_to_predict1=pad_sequences(pos_to_predict,maxlen=50,padding="post", value=len(list_of_pos))

	D=np.reshape(D1,(D1.shape[0],50,1))
	P=np.reshape(P1,(P1.shape[0],50,1))
	n_words=len(word_index)

	D_to_predict=np.reshape(D_to_predict1,(D_to_predict1.shape[0],50,1))
	P_to_predict=np.reshape(P_to_predict1,(P_to_predict1.shape[0],50,1))	

	embedding_matrix = np.zeros((n_words, 300))

	for word,i in word_index.items():
		if(i>=len(word_index)):
			continue
		if word in glove_emb:
			embedding_matrix[i]=glove_emb[word]

	tag_list=['negative','positive','neutral']#,'neutral','conflict']
	n_tags=len(tag_list)

	max_len=50
	tag2idx={t:i for i,t in enumerate(tag_list)}

	y = [tag2idx[s[1]] for s in sentences]
	asp=[s[2] for s in sentences]
	kk_asp=[s[1] for s in kk_sentences]
	kk_sents=[s[2] for s in kk_sentences]
	sents=[s[3] for s in sentences]
	y = [to_categorical(i, num_classes=n_tags) for i in y]

	validation_size=int(0.2*X.shape[0])
	
	X_tr=X[:-validation_size]
	y_tr=y[:-validation_size]
	asp_tr=asp[:-validation_size]
	sents_tr=sents[:-validation_size]
	D_tr=D[:-validation_size]
	P_tr=P[:-validation_size]
	X_te=X[-validation_size:]
	y_te=y[-validation_size:]
	D_te=D[-validation_size:]
	P_te=P[-validation_size:]
	asp_te=asp[-validation_size:]
	sents_te=sents[-validation_size:]

	print(X_tr.shape,np.asarray(y_tr).shape)

	vocab_size=len(word_index)
	inputt= Input(shape=(50,))
	emb=Embedding(vocab_size,300,weights=[embedding_matrix],
							input_length=50,
							#mask_zero=True,
							trainable=False)(inputt)

	other_features=Input(shape=(50,1))
	other_features1=Input(shape=(50,1))
	emb=merge([emb,other_features,other_features1],mode='concat')

	gru_f=Bidirectional(GRU(50,return_sequences=True))(emb)
	dense=Dense(25,activation='relu')(gru_f)
	drop=Dropout(0.1)(dense)
	drop=Flatten()(drop)
	out=Dense(3,activation='softmax')(drop)

	model=Model([inputt,other_features,other_features1],out)

	model.compile(loss='categorical_crossentropy',
              optimizer="rmsprop",
              metrics=['accuracy'])

	model.fit([X,D,P], np.array(y), batch_size=25, epochs=5, validation_split=0.1, verbose=0)

	#print(model.evaluate([X_te,D_te,P_te], np.array(y_te)))

	k=model.predict([X_to_predict,D_to_predict,P_to_predict])
	print(kk_sents[0])
	ret_val=[]
	for i in range(0,len(k)):
		p=np.argmax(k[i],axis=-1)
		ret_val.append((kk_asp[i].lower(),tag_list[p]))
		#print(kk_asp[i].lower(),tag_list[p])
	return ret_val