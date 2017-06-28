
from nltk import word_tokenize
import os
import random
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

from sklearn.model_selection import train_test_split, cross_val_predict
import simplejson as json
import re
import codecs
from Classifiers import MutableMLPClassifier,MutableKNeighborsClassifier, MutableLinearSVC,MutablePassiveAggressiveClassifier,MutableRandomForestClassifier,MutableSGDClassifier



stopwords_list_path = 'D:/Sasha/subversion/stopwrds.txt'#words from metadata of texts
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

        auc_out = open(outpath['AUC'], 'a')
        auc_out.write("('"+classifier['name']+"', " + str(AUC) + '), ')
        auc_out.close()

        acc_out = open(outpath['ACC'], 'a')
        acc_out.write("('"+classifier['name']+" '," + str(A) + '), ')
        acc_out.close()

        pr_out = open(outpath['PR'], 'a')
        pr_out.write("('"+classifier['name']+"', " + str(Macro_P) + '), ')
        pr_out.close()

        rc_out= open(outpath['RC'], 'a')
        rc_out.write("('"+classifier['name']+"', " + str(Macro_R) + '), ')
        rc_out.close()

        f_out = open(outpath['F'], 'a')
        f_out.write("('" +classifier['name']+" ', " + str(Macro_F) + '), ')
        f_out.close()


path_to_config = 'D:/Sasha/subversion/trunk/AuthorshipAttributionRussianTexts/config.json'
settings=[]
with codecs.open(path_to_config, 'r') as f:
    settings = f.read()
    settings = json.loads(settings)

for d in settings['data']:
    all_data = LooadData(d['path'])

    all_data['authors'] = ReplaseAuthorsNamesWithNumeric(all_data['authors'])
    splited_data = SplitData(all_data,20,0.6)
    train_data = splited_data['train']
    test_data = splited_data['test']

    #creating classifiers
   # mutableKNeighborsClassifier =  MutableKNeighborsClassifier(k=d['k_features']['KNN'])
    mutableLinearSVC = MutableLinearSVC(k=d['k_features']['LinearSVC'])
    mutablePassiveAggressiveClassifier = MutablePassiveAggressiveClassifier(k=d['k_features']['PassiveAggressive'])
   # mutableRandomForestClassifier = MutableRandomForestClassifier(k=d['k_features']['RandomForest'])
    mutableSGDClassifier = MutableSGDClassifier(k=d['k_features']['SGD'])

    composite_classifier = VotingClassifier(estimators=[
        ('Linear Support Vector Classification', mutableLinearSVC), ('Passive Aggressive Classifier', mutablePassiveAggressiveClassifier),
        ('Stochastic gradient descent classifier', mutableSGDClassifier)
    ], voting='hard', weights=[1,  1, 1])


    classifiers = [
        #{"name":"MLPC classifier",
        # "clf": MutableMLPClassifier(k=d['k_features']['MLPC'])},
      #  {"name":"KNN",
      #   "clf": mutableKNeighborsClassifier},
        {"name":"LSV",
         "clf": mutableLinearSVC},
        {"name":"PA",
         "clf": mutablePassiveAggressiveClassifier},
       # {"name":"RF",
       #  "clf": mutableRandomForestClassifier},
        {"name":"SGD",
         "clf": mutableSGDClassifier},
        {"name":"COMP",
         "clf": composite_classifier}
    ]

    vectorizer = TfidfVectorizer(min_df=2,
                                 #ngram_range=[2,2],
                                 max_df = 0.8,
                                 stop_words= stopwords,
                                 sublinear_tf=True,
                                 use_idf=True,
                                 lowercase=True)

    #vectorize our  data
    train_data = VectorizeData(train_data,vectorizer)

    classifiers = TrainClassifiers(classifiers, train_data)

    outpath = settings['output']['path']
    testdata = []
    for author, text in zip(test_data['authors'],test_data['texts']):
        dataOb = {'author':author, 'text': text}
        testdata.append(dataOb)

    for n in range(1,202):
        random.shuffle(testdata)
        texts =[]
        authors = []
        for dataobj in testdata:
            texts.append(dataobj['text'])
            authors.append(dataobj['author'])

        test_data['authors'] = authors
        test_data['texts'] = texts

        test_data_random = SplitData(test_data,20,0.8)
        test_data_random = test_data_random['train']

        X_test = vectorizer.transform(test_data_random['texts'])
        test_data_random['texts'] = X_test

        PredictAndShowResult(classifiers,test_data_random)



