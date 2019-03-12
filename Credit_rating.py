# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 16:25:11 2019

@author: gaura
"""

from AdvancedAnalytics import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
import pandas as pd
import numpy as np
#importing the dataset 
df=pd.read_excel("CreditHistory_Clean.xlsx")
#encoding using a self defined function
def my_encoder(z):
    for i in z:
        a=df[i][df[i].notnull()].unique()
        for col_name in a:
            df[i+'_'+str(col_name)]= df[i].apply(lambda x: 1 if x==col_name else 0)
    
    
categorical = ['checking','coapp','depends','employed','existcr','foreign','history','housing','installp','job','marital','other','property','purpose','resident','savings','telephon']

my_encoder(categorical)
df= df.drop(columns=categorical)
X= np.asarray(df.drop(columns="good_bad"))
Y= df["good_bad"]
Y= Y.map({"good":1,"bad":0})
Y = np.asarray(Y)

from sklearn.model_selection import cross_val_score
score_list=['recall','accuracy','precision','f1']
search_depths=[5,6,7,8,10,12,15,20,25]

#using cross validation to select the best depth from a list of depths for a decision tree 
Table = pd.DataFrame(index=range(9),columns= score_list)
k=0
for d in search_depths:
    dtc= DecisionTreeClassifier(criterion="gini", max_depth=d, min_samples_split=5,min_samples_leaf=5)
    mean_score=[]
    std_score=[]
    for s in score_list:
        dtc_10 = cross_val_score(dtc,X,Y,scoring=s, cv=10)
        mean= dtc_10.mean()
        Table.loc[k,s] = mean
    k=k+1

#best model is for depth=5

from sklearn.model_selection import train_test_split

#splitting into train and test sets
X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3) 

#Model training usinf the training data and displaying binary metrics
dtc_train= DecisionTreeClassifier(criterion="gini", max_depth=5,  min_samples_leaf=5,min_samples_split=5)
dtc_train_fit = dtc_train.fit(X_train,Y_train)
DecisionTree.display_binary_metrics(dtc_train_fit,X_train,Y_train)

#binary Metrics for test data
DecisionTree.display_binary_metrics(dtc_train_fit,X_test,Y_test)

#Displaying the tree
dot_data = export_graphviz(dtc_train,filled=True,rounded=True,out_file=None)
graph_png = graph_from_dot_data(dot_data) 
import graphviz 
graph_png.write_png('Decision_tree.png') 


 
 
