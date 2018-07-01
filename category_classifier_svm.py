import numpy as np
import nltk,random
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, cross_val_score
# from autocorrect import spell
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

from collections import defaultdict
D=defaultdict(list)
L=defaultdict(list)
cls={}
def classification1(data,labels,labels_polar,kk,test_labels,test_labels_polar):
	for i in range(0,len(data)):
		for j in range(0,len(labels[i])):
			D[labels[i][j]].append(data[i])
			L[labels[i][j]].append(labels_polar[i][j])

	mlb = MultiLabelBinarizer()
	y=mlb.fit_transform(labels)
	y_test=mlb.transform(test_labels)
	print(mlb.classes_)
	# vectorizer = TfidfVectorizer(min_df=1,ngram_range=(1, 3))
	classifier = Pipeline([('tfidf', TfidfVectorizer(min_df=1,ngram_range=(1, 1))),('clf', OneVsRestClassifier(LinearSVC()))])
	# scores=cross_val_score(classifier,data,y,cv=5,scoring="f1_samples")
	# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	
	for i in D.keys():
		cls[i] = Pipeline([('tfidf', TfidfVectorizer(min_df=1,ngram_range=(1, 1))),('clf', OneVsRestClassifier(LinearSVC()))])
		cls[i].fit(D[i],L[i])

	classifier.fit(data,y)
	te_pred=classifier.predict(kk)
	print("FSCORE",f1_score(te_pred,y_test,average='samples'))
	
	cat=[]
	for i in kk:
		f=mlb.inverse_transform(classifier.predict([i]))[0]
		if(len(f)==0):
			f=(u'anecdotes/miscellaneous',)
		cat.append(f)
	#print(cat)
	#print(kk)
	ret_val=[]
	for i in range(0,len(cat)):
		temp=[]
		for j in cat[i]:
			temp.append((j,cls[j].predict([kk[i]])[0]))
		ret_val.append(temp)
		#print(i,cls[i].predict([kk])[0])
	return ret_val
