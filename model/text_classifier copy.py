import numpy as np
import re
import nltk
from sklearn.datasets import load_files

#movie_data_aug = load_files(r".\dataset_aug\syn")
#X_aug, y_aug = movie_data_aug.data, movie_data_aug.target

movie_data_train = load_files(r".\dataset\minibatch")
X, y = movie_data_train.data, movie_data_train.target
leng = len(X)
mi = leng/2
X_woa= X[:20]
print(X_woa)
