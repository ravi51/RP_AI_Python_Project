#### Model_App.py file #####
#### Created on 2024-06-10
#### Created by: Ravindra Patole
#### Modified on 
#### Modified by: 
#### Description: This file is used to create, Train and test the model for Language Detection using SVM Classifier.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import re

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# remove warnings at the time of execution 
import warnings
warnings.filterwarnings('ignore')

#create function to remove special characters and digits from the text data


def clean_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert to lowercase
    text = text.lower()
    return text


# read the CSV file 
print("Reading the CSV file for Language Detection")
df=pd.read_csv(r"E:\AI_Projects\NLP_Projects\Language_detection\Data\Language Detection.csv")
print("First 5 rows of the dataset:")
print(df.head(5))

print("Columns in the dataset:\n",df.columns)

#get the count of each language in the dataset
print("Count of each language in the dataset:")
print(df['Language'].value_counts())


#apply the clean text function to remove special characters and digits from the 'Text' column
df['Cleaned_Text'] = df['Text'].apply(clean_text)

print("First 5 rows of the cleaned dataset:")
print(df.head(5))


# convert the text data into numerical features using CountVectorizer
# it is based on word counts and creates a sparse matrix representation of the text data
vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3))
xc = vectorizer.fit_transform(df['Cleaned_Text'] )
yc =df['Language']
with open(r'E:/AI_Projects/NLP_Projects/Language_detection/Models/Cvectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

#Convert the text data into numerical features using TF-IDF vectorization
#it is based on importance of words in the text data and creates a sparse matrix representation of the text data

tvectorizer = TfidfVectorizer(analyzer='char',ngram_range=(1,3))
xt =tvectorizer.fit_transform(df['Cleaned_Text'])
yt =df['Language']

# save the vectorize into pkl file for future use
with open(r'E:/AI_Projects/NLP_Projects/Language_detection/Models/Tfvectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Divide data into training and testing purposes using  Train_test_Split function
#TfidfVectorizer
X_traint, X_testt, y_traint, y_testt =train_test_split (xt, yt ,test_size=0.2 ,random_state=42)
#CountVectorizer
X_trainc, X_testc, y_trainc, y_testc =train_test_split (xc, yc ,test_size=0.2 ,random_state=42)

####################    MultinomialNB       #######################
# used Multinomial naive_bayes algorithm 
# to train the model and predict the language of the text data
nb_model =MultinomialNB()

# Train data Fit in the  model
#nb_model.fit(X_traint,y_traint)

nb_model.fit(X_trainc,y_trainc)

#predict X-test data using the trained model
#nb_pred = nb_model.predict(X_testt)

nb_pred = nb_model.predict(X_testc)

print(" ****  Accuracy of the model NB **** \n")
#print(round(nb_model.score(X_testt,y_testt),2))
print(round(nb_model.score(X_testc,y_testc),2))

print("****  Classification Report of the model **** \n")
#print(classification_report(y_testt,nb_pred))
print(classification_report(y_testc,nb_pred))
# save the nb_model into pkl file for future use
with open(r'E:/AI_Projects/NLP_Projects/Language_detection/Models/nb_model.pkl', 'wb') as file:
    pickle.dump(nb_model, file)


######### LogisticRegression #############

lr_model =LogisticRegression()
lr_model.fit(X_traint,y_traint)
y_predlr =lr_model.predict(X_testt)


print(" ****  Accuracy of the model  LR**** \n")
print(round(lr_model.score(X_testt,y_testt),2))
print("****  Classification Report of the model  LR**** \n")
print(classification_report(y_testt,y_predlr))

with open(r'E:/AI_Projects/NLP_Projects/Language_detection/Models/lr_model.pkl', 'wb') as file:
    pickle.dump(lr_model, file)