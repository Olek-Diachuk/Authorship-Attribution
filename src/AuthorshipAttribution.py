import itertools
from nltk import word_tokenize
import os
import sys
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_predict
from texttable import Texttable
import simplejson as json
import re
import codecs
from Classifiers import MutableKNeighborsClassifier, MutableLinearSVC,MutablePassiveAggressiveClassifier,MutableRandomForestClassifier,MutableSGDClassifier

import matplotlib.pyplot as plt


stopwords_list_path = 'D:/Sasha/subversion/new4.txt'
with codecs.open (stopwords_list_path, "r", encoding='utf-8') as myfile:
    stops = myfile.read()
stopwords = word_tokenize(stops)


def ReplaseAuthorsNamesWithNumeric(y_test_to_replace):
    i=1
    string_y_test = ','.join(y_test_to_replace)
    for author in set(y_test_to_replace):
        string_y_test = string_y_test.replace(author, str(i))
        i+=1
    string_y_test=map(int, string_y_test.split(','))
    return string_y_test

def LooadData(path):
    data = {}
    pattern_author_ru = '([a-zA-Z]*)'
    all_data = []
    all_labels = []
    for fname in os.listdir(path):
        fileObj = codecs.open( os.path.join(path, fname), "r", "utf_8_sig" )
        text = fileObj.read()
        author = re.search(pattern_author_ru, fname).group(1)
        all_data.append(text)
        all_labels.append(author)
    data['texts'] = all_data
    data['authors']= all_labels
    return data

def SplitData(data, random_state, train_size):
    splited_data={}
    X_train, X_test, y_train, y_test = train_test_split(data['texts'], data['authors'], random_state = random_state ,train_size = train_size)
    splited_data['train']={}
    splited_data['test']={}
    splited_data['train']['texts'] = X_train
    splited_data['train']['authors'] = y_train
    splited_data['test']['texts'] = X_test
    splited_data['test']['authors'] = y_test
    return  splited_data

def TrainClassifiers(classifiers, train_data):
    for clf in classifiers:
        clf['clf'].fit(train_data['texts'],train_data['authors'])
    return classifiers

def VectorizeData(train_data,vectorizer):
    X_train = vectorizer.fit_transform(train_data['texts'])
    train_data['texts'] = X_train
    return train_data

def PredictAndShowResult(classifiers,test_data):
    y_test = test_data['authors']
    X_test = test_data['texts']
 

    for classifier in classifiers:
        clf = classifier['clf']
        predicted = cross_val_predict(clf, X_test, y_test, cv=10)
        #predicted = clf.predict(X_test)
        Macro_P = (metrics.precision_score(y_test, predicted, average='macro')*100)
        A = (metrics.accuracy_score(y_test, predicted)*100)
        Macro_R = (metrics.recall_score(y_test, predicted, average='macro')*100)
        Macro_F = (metrics.f1_score(y_test, predicted, average='macro')*100)
        Micro_P = (metrics.precision_score(y_test, predicted, average='micro')*100)
        Micro_R = (metrics.recall_score(y_test, predicted, average='micro')*100)
        Micro_F = (metrics.f1_score(y_test, predicted, average='micro')*100)
        fpr, tpr, thresholds = metrics.roc_curve(y_test , predicted, pos_label=16)
        AUC = (metrics.auc(fpr, tpr)*100)
        ######################### Presenting results ###################################################
        table = Texttable()
        print (classifier['name']+' measures: ')
        table.add_rows([['Measure','Result'],
                        ['ACCURACY ', A],
                        ['Error Rate', 100-A],
                        ['The MACRO Precission ', Macro_P],
                        ['The MACRO Recall ', Macro_R],
                        ['The MACRO FScore ', Macro_F],
                        ['The MICRO Precission ', Micro_P],
                        ['The MICRO Recall ', Micro_R],
                        ['The MICRO FScore ', Micro_F],
                        ['Area Under Curve', AUC]
                        ])
        print (table.draw())
        print ('\n Detailed classification report for '+classifier['name']+':')
        print (classification_report(y_test, predicted))
        #################################### Confusion Matrix ##########################################

        def plot_confusion_matrix(cm, classes,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues):
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            cmRound = []
            print(cm)
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, cm[i, j],
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.ylabel('True authors')
            plt.xlabel('Predicted authors')
        # Compute confusion matrix
        conf_m = confusion_matrix(y_test, predicted)
        np.set_printoptions(precision=2)
        # Plot normalized confusion matrix for emotions
        plt.figure()
        plot_confusion_matrix(conf_m, classes=set(y_test_old),
                              title=features_name+'. Classifier name: '+classifier['name'])
       # fname = features_name+'_'+classifier['name']
        #fpath = 'D:/Sasha/subversion/trunk/AuthorshipAttributionRussianTexts/results/conf_matrix/'+fname #to save in file
        plt.show()

path_to_config = 'D:/Sasha/subversion/trunk/AuthorshipAttributionRussianTexts/config.json'
filess=[]
with codecs.open(path_to_config, 'r') as f:
    filess = f.read()
    filess = json.loads(filess)

for d in filess['data']:
    all_data = LooadData(d['path'])
    y_test_old =  all_data['authors']
    all_data['authors'] = ReplaseAuthorsNamesWithNumeric(all_data['authors'])
    splited_data = SplitData(all_data,20,0.6)
    train_data = splited_data['train']
    test_data = splited_data['test']

    #creating classifiers
    mutableKNeighborsClassifier =  MutableKNeighborsClassifier(k=d['k_features']['KNN'])
    mutableLinearSVC = MutableLinearSVC(k=d['k_features']['LinearSVC'])
    mutablePassiveAggressiveClassifier = MutablePassiveAggressiveClassifier(k=d['k_features']['PassiveAggressive'])
    mutableRandomForestClassifier = MutableRandomForestClassifier(k=d['k_features']['RandomForest'])
    mutableSGDClassifier = MutableSGDClassifier(k=d['k_features']['SGD'])

    composite_classifier = VotingClassifier(estimators=[
        ('Linear Support Vector Classification', mutableLinearSVC), ('Passive Aggressive Classifier', mutablePassiveAggressiveClassifier),
        ('Stochastic gradient descent classifier', mutableSGDClassifier)
    ], voting='hard', weights=[2,  1, 2])


    classifiers = [
                    #{"name":"MLPC classifier",
                   # "clf": MutableMLPClassifier(k=d['k_features']['MLPC'])},
                 #  {"name":"K-nearest neighbors Classifier",
                #    "clf": mutableKNeighborsClassifier},
                   {"name":"Linear Support Vector Classification",
                    "clf": mutableLinearSVC},
                   {"name":"Passive Aggressive Classifier",
                    "clf": mutablePassiveAggressiveClassifier},
                   {"name":"Random Forest Classifier",
                    "clf": mutableRandomForestClassifier},
                   {"name":"Stochastic gradient descent classifier",
                    "clf": mutableSGDClassifier},
                   {"name":"Composite classifiers",
                    "clf": composite_classifier}
                   ]



    vectorizer = TfidfVectorizer(min_df=2,
                    #ngram_range=[5,5],
                    max_df = 0.8,
                  #  analyzer ='char',
                    stop_words= stopwords,
                    sublinear_tf=True,
                    use_idf=True,
                    lowercase=True)


    #vectorize our train data
    train_data = VectorizeData(train_data,vectorizer)
    # we want to split TEST data and pass it no classifier only 80%
    test_data_random = SplitData(test_data,20,0.8)
    test_data_random = test_data_random['train']
    X_test = vectorizer.transform(test_data_random['texts'])
    test_data_random['texts'] = X_test

    classifiers = TrainClassifiers(classifiers, train_data)

    with open("D:/Sasha//log_result1.txt", "a") as  out:
        sys.stdout = out
        features_name = d['features_name']
        print ('='*40)
        print ('='*40)
        print (features_name.encode('utf-8'))
        PredictAndShowResult(classifiers,test_data_random)



