import sys
import json
import numpy as np
# import tensorflow as tf
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn import svm
import random


def print_request(r):
    """Print a request in a human readable format to stdout"""
    def fmt_token(t):
        return t['shape'] + t['after']

    print('Subject: ' + ''.join(map(fmt_token, filter(
        lambda x: x['where'] == 'subject', r['tokens']))))
    print(''.join(map(fmt_token, filter(
        lambda x: x['where'] == 'body', r['tokens']))))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("predict_category <train_file> <test_file>")

    train_file, test_file = sys.argv[1:]
    

    # ToDo: Implement logic to find a model based on data in train_file
    train_file, test_file = sys.argv[1:]
    print("Loading training File ",train_file)
    data =json.load(open(train_file))
    
    #get Lables 
    labels_count={}
    for i in range(len(data)):
        for label in data[i]['labels'].keys():
            if label in labels_count:
                labels_count[label]+=1
            else:
                labels_count[label]=1
    lables_ids={}
    for lab,itr  in zip(labels_count.keys(),range(len(labels_count.keys()))):
        lables_ids[lab]=itr
        
    #NER tokens -> required for feature mapping
    ner={}
    for i in range(len(data)):
        for token in data[i]['tokens']:
            ner[token['rner']]=1   
    nerFeaturePosition={}
    for value,pos in zip(ner.keys(),range(len(ner.keys()))):
        nerFeaturePosition[value]=pos
        

    #features 
    #where_1 = is_body
    #where_2 = is_subject
    #shape_1 = begins_with_capital
    #shape_2 = contains_colon
    #shape_3 = contains_hyphen
    #shape_4 = contains_d
    #start   = is_begining (is 1 if its positon is less than 10)
    #ner = 2 placeholder for every nerType (so 2*24) + 1 for other
    #feature vector for every token = 2+4+1+49 = 56

    maxTokenLength = 2600 #required for padding  If text contains more than 2600 tokens, it will be ignored. if less thant it ll be padded with 0
    tokenFeaturesCount=56

    def featureMapping(tokens):
        featureVector = np.zeros([maxTokenLength,tokenFeaturesCount])

        ##Feature Vector is used for CNN -> which I wanted to experiment

        columns=[]
        for tok,itr in zip(tokens,range(len(tokens))):
            if itr>= maxTokenLength:
                break
            if tok['where']=='body':
                featureVector[itr][0]=1
                columns.append(itr*tokenFeaturesCount+0)

            if tok['where']=='subject':
                featureVector[itr][1]=1
                columns.append(itr*tokenFeaturesCount+1)

            if tok['shape'].startswith('X'):
                featureVector[itr][2]=1
                columns.append(itr*tokenFeaturesCount+2)

            if ':' in tok['shape']:
                featureVector[itr][3]=1
                columns.append(itr*tokenFeaturesCount+3)

            if '-' in tok['shape']:
                featureVector[itr][4]=1
                columns.append(itr*tokenFeaturesCount+4)

            if 'd' in tok['shape']:
                featureVector[itr][5]=1
                columns.append(itr*tokenFeaturesCount+5)

            if tok['start'] < 10:
                featureVector[itr][6]=1
                columns.append(itr*tokenFeaturesCount+6)

            nerFeature = 7+int(nerFeaturePosition[tok['rner']])
            columns.append(itr*tokenFeaturesCount+nerFeature)
            featureVector[itr][nerFeature]=1

        return columns
    
    random.shuffle(data)
    splitPoint=int(len(data)*0.75)
    trainingData = data[:splitPoint]
    validationData=data[splitPoint:]
    
    def dataToFeature(data,testSet=False):
        colm=[]
        row = []
        d =[]
        tar=[]
        batch=len(data)
        print("Genearating feature vector ",batch)
        for i in range(len(lables_ids)):
            tar.append(np.zeros(batch))

        for i in range(batch):
            if i %1500==0:
                print (i," loaded")
            fea=featureMapping(data[i]['tokens'])
            for ele in fea:
                colm.append(ele)
                row.append(i)
                d.append(1)

            if testSet == True:
                continue
            for lab in data[i]['labels']:
                tar[lables_ids[lab]][i]=1

        #dummy variable to make feature length of test/validation/train same size
        row.append(i)
        d.append(0)
        colm.append(maxTokenLength*tokenFeaturesCount)
        return sparse.csr_matrix((d,(row,colm))),tar
    
    feature_valid,label_valid=dataToFeature(validationData)
    print("Validation data")
    feature_train,label_train=dataToFeature(trainingData)
    models ={}
    yPred={}
    for label in lables_ids:
        labelId = lables_ids[label]
        print("Buidling model for ",label,labelId)
        ##Cross Validation and Hyper parameter tuning
            #     parameter_candidates = [
        #   {'C': [1, 4, 16, 32,64,1024], 'kernel': ['linear']}
        # ]
        #     clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=3,scoring=f1_scorer)
        #     clf.fit(feature_train,label_train[labelId])
        models[label] = svm.SVC(kernel='linear', C = 1.0,probability=True,verbose=True)
        models[label].fit(feature_train,label_train[labelId])
    
    print("Before threshold tuning ")
    yPred={}
    yPred_prob={}
    for label in lables_ids:
        labelId = lables_ids[label]
        yPred[label]=models[label].predict(feature_valid)
        yPred_prob[label]=models[label].predict_proba(feature_valid)
        print(label," F-Score " ,f1_score(label_valid[labelId],yPred[label]))

    print("After threshodl tuning")
    bestThreshold={}
    for label in lables_ids:
        labelId=lables_ids[label]
        bestThreshold[label]=0.5
        bestFScore=0
        thresSorted =sorted(set([round(x,3) for x in sorted(yPred_prob[label][:,1])]))
        for thres in thresSorted:
            tempFscore=f1_score(label_valid[labelId],yPred_prob[label][:,1]>thres)
            if tempFscore>bestFScore:
                bestFScore=tempFscore
                bestThreshold[label]= thres
        print(label," Best F-score ",bestFScore," Optimal threshold thres",bestThreshold[label])
        
     

    # ToDo: Make predictions on data in test_file
    print("Loading Test data")
    data_test = json.load(open(test_file))

    # print("Test Data Feature gen")
    feature_test,temp=dataToFeature(data_test,testSet=True)
    print("Predicting Labels")
    yPred_test={}
    yPred_prob_test={}
    for label in lables_ids:
        labelId = lables_ids[label]
        yPred_test[label]=models[label].predict(feature_test)
        yPred_prob_test[label]=models[label].predict_proba(feature_test)
    
    # ToDo: Generate output
    for i in range(len(data_test)):
        data_test[i]['labels']={}
        noneLabel=True
        for label in lables_ids:
            labelId = lables_ids[label]
    #         print(label,yPred_prob_test[label][i][1],bestThreshold[label])
            if yPred_prob_test[label][i][1]>bestThreshold[label]:
                noneLabel=False
                data_test[i]['labels'][label]=yPred_prob_test[label][i][1]
        if noneLabel :
            data_test[i]['labels']['others']=1
    #     break
    
    print("comtravo_challenge_test.json has been created")
    json.dump(data_test,open("comtravo_challenge_test.json","w"))
