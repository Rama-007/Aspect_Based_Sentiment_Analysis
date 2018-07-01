

from collections import defaultdict
ontology_dict=defaultdict(list)
noun_frequency=defaultdict(int)
category_list=[]
import nltk





def ontology(data,labels):
	for e1 in labels:
		category_list.extend(e1)
	category_list1=list(set(category_list))
	for text in data:
		words=nltk.word_tokenize(text)
		pos=nltk.pos_tag(words)
		for word in pos:
			if('NN' in word[1]):
				noun_frequency[word[0]]+=1

	for i in range(0,len(data)):
		categories=labels[i]
		words=nltk.word_tokenize(data[i])
		pos=nltk.pos_tag(words)
		for word in pos:
			if('NN' in word[1]):
				if(noun_frequency[word[0]]>=5):
					for category in categories:
						if(word[0] not in ontology_dict[category]):
							ontology_dict[category].append(word[0])


	# print(ontology_dict)

	return ontology_dict

	