

import nltk
from collections import defaultdict
from scipy import spatial
import numpy as np


text_aspects=[]

fp=open("glove.6B/glove.6B.300d.txt","r")
glove_emb={}
for line in fp:
	temp=line.split(" ")
	glove_emb[temp[0]]=np.asarray([float(i) for i in temp[1:]])

ontology_embeded=defaultdict(list)

def extractor(ontology_dict,data,labels,aspects):
	for i in ontology_dict.keys():
		listt=ontology_dict[i]
		for j in listt:
			try:
				if(glove_emb[j] not in ontology_embeded[i]):
					ontology_embeded[i].append(glove_emb[j])
			except:
				pass
	found_aspect=[]
	for i in range(0,len(data)):
		words=nltk.word_tokenize(data[i])
		pos=nltk.pos_tag(words)
		temp=[]
		for word in pos:
			if('NN' in word[1]):
				temp.append(word[0])

		categories=labels[i]
		temp_aspect=[]
		for word in temp:
			try:
				vec=glove_emb[word]
			except:
				continue
			mini=0
			for category in categories:
				for embeddings in ontology_embeded[category]:
					mini=1-spatial.distance.cosine(vec,embeddings)
					if(mini>0.6):
						temp_aspect.append((category,word))
						break
		# print(temp_aspect)
		found_aspect.append(temp_aspect)


	for i in range(0,len(data)):
		print(aspects[i],found_aspect[i])
		if(i>10):
			break


