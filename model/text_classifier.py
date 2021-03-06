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

import nlpaug
import sys
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

#movie_data_aug = load_files(r".\dataset_aug\syn")
#X_aug, y_aug = movie_data_aug.data, movie_data_aug.target

movie_data_train = load_files(r".\dataset\minibatch")
X, y = movie_data_train.data, movie_data_train.target

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

    #aug
    aug = naw.SynonymAug()
    augmented_text = aug.augment(document)
    print('augtext', augmented_text)
    document_aug = augmented_text
    #y=np.insert(y,y[sen],1)
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    if sen < 20:
        documents.append(document)
    else:
        y = y[:len(X)-24]

    

#X_woa, y_woa = movie_data_train.data, movie_data_train.target
#X_woa, y_woa = list(range(1,20))
#print('new data excluding augmentation', X_woa)


vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()
#X_woa = vectorizer.fit_transform(documents_t).toarray()

tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()
#X_woa = tfidfconverter.fit_transform(X_woa).toarray()

tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(documents).toarray()
#X_woa = tfidfconverter.fit_transform(documents_t).toarray()

X_woa,y_woa = X[20:], y[20:]
#X_train, X_test, y_train, y_test = train_test_split(X_woa, y_woa, test_size=0.2, random_state=0)
#use test split whole data without aug
#use train with aug

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X, y) 

y_pred = classifier.predict(X_woa)

print(confusion_matrix(y_woa,y_pred))
print(classification_report(y_woa,y_pred))
print(accuracy_score(y_woa, y_pred))

with open('text_classifier', 'wb') as picklefile:
    pickle.dump(classifier,picklefile)
#with open('text_classifier', 'rb') as training_model:
 #   model_org = pickle.load(training_model)