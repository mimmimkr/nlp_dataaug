import nlpaug
import sys
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

data = load_files(r'.\dataset\trainingset')
X, y = data.data, data.target
dataaugtype = 'basic'
print (X[1])
print('sds')
print(data.filenames[1])
sentences = []
def basic_aug(i, augtype):
    augsentences = []
    print(len(X[i].decode('utf-8')))
    sentences= X[i].decode('utf-8').split('\n')

    #data aug on first review
        #aug1 = nac.KeyboardAug()
    if (augtype == 'basic'):
        aug = nac.KeyboardAug()
    if(augtype == 'cwe'):
        aug = naw.ContextualWordEmbsAug()
    if(augtype == 'rndw_del'):
        aug = naw.RandomWordAug(action='delete')
    if(augtype == 'ocr'):
        aug = nac.OcrAug()
    if(augtype == 'rndw_swap'):
        aug = naw.RandomWordAug(action='swap')
    if(augtype == 'syn'):
        aug = naw.SynonymAug()
    if(augtype == 'spell'):
        aug= naw.SpellingAug()
    for x in range(len(sentences)):
        print(sentences[x])
        augsentence = aug.augment(sentences[x])
        augsentences.append(augsentence)
        print(augsentences)
    augmented_text = aug.augment(X[i].decode('utf-8'))
        
    #print("Original:")
    #print(X[i].decode('utf-8'))
    #print("Augmented Text:")
    #print(augmented_text)
    return ' '.join(augsentences)


def write_vars_to_file(augfolder):
    for af in range(len(X)):
        #len(X)
        #get info from current file
        filepath = data.filenames[af].split('\\')
        classname = filepath[3]
        filename = filepath[4]
        with open('./dataset_aug/'+augfolder+'/'+classname+'/'+'_aug'+filename, "w") as file:
           file.write(basic_aug(af, augfolder))
    
#write_vars_to_file('basic')
write_vars_to_file('ocr')