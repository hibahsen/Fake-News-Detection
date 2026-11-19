import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Read the data
df=pd.read_csv('fakeNews\WELFake_Dataset.csv.zip')

df["label"] = df["label"].map({0: "FAKE", 1: "REAL"})

df["text"] = df["text"].fillna("")

#Get shape and head
df.shape
df.head()

#DataFlair - Get the labels
labels = df.label
labels.head()

#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)

#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])

#testing on a new file

# Read the new articles
new_articles = pd.read_csv("fakeNews/new_articles.csv")

# Replace any missing text with an empty string
new_articles["text"] = new_articles["text"].fillna("")

# Transform the articles using the trained TF-IDF vectorizer
new_vectors = tfidf_vectorizer.transform(new_articles["text"])

# Predict
predictions = pac.predict(new_vectors)

print("\ndetecting fake news...")
print(f"Model Test Accuracy: {round(score*100,2)}%\n")

for i, prediction in enumerate(predictions):
    print(f"Article {i+1}")
    
    # Print the title if there is one
    if "title" in new_articles.columns:
        print(f"Title: {new_articles.iloc[i]['title']}")

    print(f"Prediction: {prediction}")
    print("")
