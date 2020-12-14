import numpy as np
import re
import nltk
from sklearn.datasets import load_files
import pickle
nltk.download('stopwords')
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import VotingClassifier

movie_data_train = load_files(r".\dataset\trainingset")
X, y = movie_data_train.data, movie_data_train.target

movie_data_test = load_files(r".\dataset\testset")
X_test, y_test = movie_data_test.data, movie_data_test.target

documents = []
documents_t = []
stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
        # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)

for sen in range(0, len(X_test)):
        # Remove all the special characters
    document_t = re.sub(r'\W', ' ', str(X_test[sen]))
    
    # remove all single characters
    document_t = re.sub(r'\s+[a-zA-Z]\s+', ' ', document_t)
    
    # Remove single characters from the start
    document_t = re.sub(r'\^[a-zA-Z]\s+', ' ', document_t) 
    
    # Substituting multiple spaces with single space
    document_t = re.sub(r'\s+', ' ', document_t, flags=re.I)
    
    # Removing prefixed 'b'
    document_t = re.sub(r'^b\s+', '', document_t)
    
    # Converting to Lowercase
    document_t = document_t.lower()
    
    # Lemmatization
    document_t = document_t.split()

    document_t = [stemmer.lemmatize(word) for word in document_t]
    document_t = ' '.join(document_t)
    
    documents_t.append(document_t)

vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()
X_test = vectorizer.fit_transform(documents_t).toarray()

tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()
X_test = tfidfconverter.fit_transform(X_test).toarray()

tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(documents).toarray()
X_test = tfidfconverter.fit_transform(documents_t).toarray()

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X, y) 

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))