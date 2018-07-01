import xml.sax
import sys
from collections import defaultdict

import nltk,random
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, cross_val_score
# from autocorrect import spell
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GRU
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.sequence import pad_sequences

# from category_classifier import classification
from category_classifier_svm import classification1

# from generate_ontology import ontology
# from aspect_extractor import extractor
#from neural_aspect_extractor import neural_extractor

from sentiment_analysis import aspect_sentiment
from final import neural_extractor1

categories=defaultdict(list)
data=[]
labels=[]
labels_polar=[]
aspects=[]
sentiments=[]

class ContentHandler(xml.sax.ContentHandler):
	def __init__(self):
		xml.sax.ContentHandler.__init__(self)
		self.flag=0
		self.data=""
		self.labels=[]
		self.labels_polar=[]
		self.aspect=[]
		self.sentiment=[]

	def startElement(self,name,attrs):
		if name == "aspectCategory":
			self.labels.append(attrs.getValue("category"))
			self.labels_polar.append(attrs.getValue("polarity"))
			categories[attrs.getValue("category")].append(self.data)
		if name == "aspectTerm":
			polar=attrs.getValue("polarity")
			if(polar=='negative' or polar=='positive'):
				self.aspect.append(attrs.getValue("term"))
				self.sentiment.append(attrs.getValue("polarity"))
		# if name == "Opinion":
		# 	self.data+=attrs.getValue("aspect")+"aspolar"+attrs.getValue("polarity")+"27071998"
		# 	# print("aspect="+ attrs.getValue("aspect"))
		# 	# print("polarity="+ attrs.getValue("polarity"))
		if name=="text":
			self.flag=1
	def endElement(self,name):
		if name=="sentence":
		# 	print (self.data)
			data.append(self.data.lower())
			labels.append(self.labels)
			labels_polar.append(self.labels_polar)
			aspects.append(self.aspect)
			sentiments.append(self.sentiment)
			self.data=""
			self.labels=[]
			self.labels_polar=[]
			self.aspect=[]
			self.sentiment=[]
		pass
	def characters(self,content):
		# pass
		if(self.flag==1):
			self.data+=content#+"27071997"
			# print(content)
			self.flag=0

source=open(sys.argv[1])
xml.sax.parse(source,ContentHandler())
print("sent for classification")
# classification(data,labels)
# classification1(data,labels)
# ontology_dict=ontology(data,labels)

# extractor(ontology_dict,data,labels,aspects)

#neural_extractor(data,labels,aspects)

#aspect_sentiment(data,aspects,sentiments)

test_size=int(0.2*len(data))
train_data=data[:-test_size]
train_labels=labels[:-test_size]
test_data=data[-test_size:]
test_labels=labels[-test_size:]
train_labels_polar=labels_polar[:-test_size]
test_labels_polar=labels_polar[-test_size:]
train_aspects=aspects[:-test_size]
test_aspects=aspects[-test_size:]

train_sentiments=sentiments[:-test_size]
test_sentiments=sentiments[-test_size:]


#text_to_predict=raw_input("ENTER:")
pred_categories=classification1(train_data,train_labels,train_labels_polar,test_data,test_labels,test_labels_polar)
print(pred_categories)
#pred_aspects=[u'food ',u'kitchen ']
pred_aspects=neural_extractor1(data,labels,aspects,test_data)
asp_acc_checker=[]
for i in range(0,len(test_aspects)):
	if(len(test_aspects[i])==0):
		asp_acc_checker.append(1)
	else:
		for j in test_aspects[i]:
			strng=''.join(pred_aspects[i])
			if(j in strng):
				asp_acc_checker.append(1)
			else:
				asp_acc_checker.append(0)

true=[1 for i in asp_acc_checker]

print(accuracy_score(asp_acc_checker,true))

print(pred_aspects)
if(len(pred_aspects)!=0):
	asp_senti=aspect_sentiment(data,aspects,sentiments,test_data,pred_aspects)
	#asp_senti=aspect_sentiment(data,aspects,sentiments,[text_to_predict],[pred_aspects])

#print("Categories, Sentiment:")
#print(pred_categories)

#print("Extracted Aspects:")
#print(test_data[:20])
#print(pred_aspects[:20])
try:
	for i in range(0,len(asp_senti)):
		print(asp_senti[i])
except:
	pass