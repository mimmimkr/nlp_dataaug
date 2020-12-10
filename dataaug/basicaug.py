import nlpaug

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split


#IMPORT DATA
movie_data = load_files(r".\dataset\txt_sentoken")
#target y, neg and pos in X
X, y = movie_data.data, movie_data.target

#divide data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("training data size: ", len(X_train))
print("test data size: ", len(X_test))

for sen in range(0,1) :
    #data aug on first review
    aug = nac.KeyboardAug()
    augmented_text = aug.augment(X_train[sen].decode('utf-8'))
    print("Original:")
    print(X_train[sen].decode('utf-8'))
    print("Augmented Text:")
    print(augmented_text)
