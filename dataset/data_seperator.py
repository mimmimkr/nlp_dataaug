import os, random
import shutil

def  copytraindata(type):
    if(type == 'n'):
        seperate = 'neg/'
    if(type == 'p'):
        seperate = 'pos/'
    path = 'nlp_dataaug/dataset/'
    data_org = os.listdir(path+'/txt_sentoken/'+seperate)

    #calc proportion/ amount for test and training set
    trainingsize=int(len(data_org)*0.8)
    orgpath = path+'/txt_sentoken/'+seperate
    destpath_train =path+'/trainingset/'+seperate
    destpath_test= path +'/testset/'+seperate

    #select files in range of trainingsize,  copy& move them to new folder
    print(trainingsize)
    rfs= []
    ofs = []
    for d in data_org:
        ofs.append(d)

    for d in range(0, trainingsize):
        rf= random.choice(data_org)
        while rf in rfs:
            #randomly selected file is aleady selected
            rf= random.choice(data_org)
        rfs.append(rf)
        shutil.copy2(orgpath+str(rf),destpath_train)

    #find the ones that are in org. but not trainingset
    for t in ofs:
        if t not in rfs: 
            print('not'+t)
            shutil.copy2(orgpath+t, destpath_test)

copytraindata('p')
copytraindata('n')