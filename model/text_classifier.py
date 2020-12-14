import numpy as np
import re
import nltk
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
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

movie_data_aug = load_files(r".\dataset_aug\syn")
X_aug, y_aug = movie_data_aug.data, movie_data_aug.target

movie_data_train = load_files(r".\dataset\txt_sentoken")
X_train, y_train = movie_data_train.data, movie_data_train.target

X,y = X_train+X_aug, y_train+y_aug

#movie_data_test = load_files(r".\dataset\testset")
#X_test, y_test = movie_data_test.data, movie_data_test.target


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

vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()
#X_test = vectorizer.fit_transform(documents_t).toarray()

tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()
#X_test = tfidfconverter.fit_transform(X_test).toarray()

tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(documents).toarray()
#X_test = tfidfconverter.fit_transform(documents_t).toarray()

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))