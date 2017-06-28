
import os
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_predict
import simplejson as json
import re
import codecs
from Classifiers import MutableKNeighborsClassifier, MutableLinearSVC,MutablePassiveAggressiveClassifier,MutableRandomForestClassifier,MutableSGDClassifier




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

    ################### Counting results for Classifiers ##########################
    i=0
    for classifier in classifiers:
        clf = classifier['clf']
        predicted = cross_val_predict(clf, X_test, y_test, cv=10)
        Macro_F = (metrics.f1_score(y_test, predicted, average='macro')*100)

        fpr, tpr, thresholds = metrics.roc_curve(y_test , predicted, pos_label=16)
        AUC = (metrics.auc(fpr, tpr)*100)
        ######################### Presenting results ###################################################

        if i == 3:
            out = open('D:/Sasha/subversion/trunk/AuthorshipAttributionRussianTexts/results/k_features/RF.txt', 'a')
        elif i == 0:
            out = open('D:/Sasha/subversion/trunk/AuthorshipAttributionRussianTexts/results/k_features/KNN.txt', 'a')
        elif i == 1 :
            out = open('D:/Sasha/subversion/trunk/AuthorshipAttributionRussianTexts/results/k_features/LSV.txt', 'a')
        elif i == 4:
            out = open('D:/Sasha/subversion/trunk/AuthorshipAttributionRussianTexts/results/k_features/SGD.txt', 'a')
        elif i == 6:
            out = open('D:/Sasha/subversion/trunk/AuthorshipAttributionRussianTexts/results/k_features/COMP.txt', 'a')
        elif i == 2:
            out = open('D:/Sasha/subversion/trunk/AuthorshipAttributionRussianTexts/results/k_features/PR.txt', 'a')

        #outputs = [knn_out,lsv_out, pr_out, rf_out, sgd_out,comp_out]
        #outputs[i].write(', '+str(Macro_F))

        out.write('  '+str(Macro_F))
        if step == 4000:
            out.write('######end feature '+ features_name.encode('utf-8')+'########')
        out.close()
        i+=1

path_to_config = 'D:/Sasha/subversion/trunk/AuthorshipAttributionRussianTexts/config.json'
settings=[]
with codecs.open(path_to_config, 'r') as f:
    settings = f.read()
    settings = json.loads(settings)
    outpath = settings['output']['path']

    for d in settings['data']:
        features_name = d['features_name']
        print ('='*40)
        print ('='*40)
        print (features_name.encode('utf-8'))

        all_data = LooadData(d['path'])
        all_data['authors'] = ReplaseAuthorsNamesWithNumeric(all_data['authors'])
        splited_data = SplitData(all_data,20,0.6)
        train_data = splited_data['train']
        test_data = splited_data['test']


        vectorizer = TfidfVectorizer(min_df=2,
                                     analyzer='word',
                                  #   ngram_range=[gram,gram],
                                     max_df = 0.8,
                                     stop_words=stopwords,
                                     sublinear_tf=True,
                                     use_idf=True,
                                     lowercase=True)
        #vectorize our train data
        train_data = VectorizeData(train_data,vectorizer)
        test_data_random = SplitData(test_data,20,0.8)
        test_data_random = test_data_random['train']
        X_test = vectorizer.transform(test_data_random['texts'])
        test_data_random['texts'] = X_test
        step = 20
        while step <4001:
            print ('='*40)
            print (features_name.encode('utf-8')+'. K = '+str(step))
            #creating classifiers
            mutableKNeighborsClassifier =  MutableKNeighborsClassifier(k=step)
            mutableLinearSVC = MutableLinearSVC(k=step)
            mutablePassiveAggressiveClassifier = MutablePassiveAggressiveClassifier(k=step)
            mutableRandomForestClassifier = MutableRandomForestClassifier(k=step)
            mutableSGDClassifier = MutableSGDClassifier(k=step)
            '''
            composite_classifier = VotingClassifier(estimators=[
                ('Linear Support Vector Classification', mutableLinearSVC), ('Passive Aggressive Classifier', mutablePassiveAggressiveClassifier),
                ('Stochastic gradient descent classifier', mutableSGDClassifier)
            ], voting='hard', weights=[1, 1, 1])
            '''
            classifiers = [
                #{"name":"MLPC classifier",
                # "clf": MutableMLPClassifier(k=d['k_features']['MLPC'])},
                {"name":"K-nearest neighbors Classifier",
                 "clf": mutableKNeighborsClassifier},
                {"name":"Linear Support Vector Classification",
                 "clf": mutableLinearSVC},
                {"name":"Passive Aggressive Classifier",
                 "clf": mutablePassiveAggressiveClassifier},
                {"name":"Random Forest Classifier",
                 "clf": mutableRandomForestClassifier},
                {"name":"Stochastic gradient descent classifier",
                 "clf": mutableSGDClassifier}#,
               # {"name":"Composite classifiers",
               #  "clf": composite_classifier}
            ]
            classifiers = TrainClassifiers(classifiers, train_data)
            PredictAndShowResult(classifiers,test_data_random)
            step += 20
