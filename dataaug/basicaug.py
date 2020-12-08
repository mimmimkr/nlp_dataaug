import nlpaug

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

from sklearn.datasets import load_files


#IMPORT DATA
movie_data = load_files(r".\dataset\txt_sentoken")
#target y, neg and pos in X
X, y = movie_data.data, movie_data.target
print(len(X))
for sen in range(0,1) :
    #data aug on first review
    aug = nac.KeyboardAug()
    augmented_text = aug.augment(X[sen].decode('utf-8'))
    print("Original:")
    print(X[sen])
    print("Augmented Text:")
    print(augmented_text)