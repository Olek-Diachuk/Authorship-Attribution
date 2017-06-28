
from sklearn import svm
from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import PassiveAggressiveClassifier


class MutableKNeighborsClassifier(KNeighborsClassifier):
    K_features = 'all'
    def __init__(self, k = 'all'):
        if k != None:
            MutableKNeighborsClassifier.K_features = k
        KNeighborsClassifier.__init__(self,n_neighbors=7, metric='cosine',algorithm='brute', weights = 'distance')

    def fit(self, *args, **kwargs):
        self.ch2 = SelectKBest(chi2, k= MutableKNeighborsClassifier.K_features)
        Y = self.ch2.fit_transform(*args, **kwargs)
        KNeighborsClassifier.fit(self,Y,args[1], **kwargs)
        return self

    def predict(self, *args, **kwargs):
        Y = self.ch2.transform(*args, **kwargs)
        return KNeighborsClassifier.predict(self,Y)

class MutableSGDClassifier(SGDClassifier):
    K_features = 'all'
    def __init__(self, k='all'):
        if k != None:
            MutableSGDClassifier.K_features = k
        SGDClassifier.__init__(self,alpha=.0001, n_iter=50,penalty="elasticnet")

    def fit(self, *args, **kwargs):
        self.ch2 = SelectKBest(chi2, k= MutableSGDClassifier.K_features)
        Y = self.ch2.fit_transform(*args, **kwargs)
        SGDClassifier.fit(self,Y,args[1], **kwargs)
        return self

    def predict(self, *args, **kwargs):
        Y = self.ch2.transform(*args, **kwargs)
        return SGDClassifier.predict(self,Y)

class MutableLinearSVC(svm.LinearSVC):
    K_features = 'all'
    def __init__(self, k='all'):
        if k != None:
            MutableLinearSVC.K_features = k
        svm.LinearSVC.__init__(self,C=2000)

    def fit(self, *args, **kwargs):
        self.ch2 = SelectKBest(chi2, k= MutableLinearSVC.K_features)
        Y = self.ch2.fit_transform(*args, **kwargs)
        svm.LinearSVC.fit(self,Y,args[1], **kwargs)
        return self

    def predict(self, *args, **kwargs):
        Y = self.ch2.transform(*args, **kwargs)
        return svm.LinearSVC.predict(self,Y)

class MutablePassiveAggressiveClassifier(PassiveAggressiveClassifier):
    K_features = 500
    def __init__(self, k):
        if k != None:
            MutablePassiveAggressiveClassifier.K_features = k
        PassiveAggressiveClassifier.__init__(self,n_iter=50)

    def fit(self, *args, **kwargs):
        self.ch2 = SelectKBest(chi2, k= MutablePassiveAggressiveClassifier.K_features)
        Y = self.ch2.fit_transform(*args, **kwargs)
        PassiveAggressiveClassifier.fit(self,Y,args[1], **kwargs)
        return self

    def predict(self, *args, **kwargs):
        Y = self.ch2.transform(*args, **kwargs)
        return PassiveAggressiveClassifier.predict(self,Y)

class MutableRandomForestClassifier(RandomForestClassifier):
    K_features = 500
    def __init__(self, k=3000):
        if k != None:
            MutableRandomForestClassifier.K_features = k
        RandomForestClassifier.__init__(self,n_estimators=40, criterion='gini')

    def fit(self, *args, **kwargs):
        self.ch2 = SelectKBest(chi2, k= MutableRandomForestClassifier.K_features)
        Y = self.ch2.fit_transform(*args, **kwargs)
        RandomForestClassifier.fit(self,Y,args[1], **kwargs)
        return self

    def predict(self, *args, **kwargs):
        Y = self.ch2.transform(*args, **kwargs)
        return RandomForestClassifier.predict(self,Y)

class MutableMLPClassifier(MLPClassifier):
    K_features = 500
    def __init__(self, k=1000):
        if k != None:
            MutableMLPClassifier.K_features = k
        MLPClassifier.__init__(self,hidden_layer_sizes=150, activation='logistic', solver='adam')

    def fit(self, *args, **kwargs):
        self.ch2 = SelectKBest(chi2, k= MutableMLPClassifier.K_features)
        Y = self.ch2.fit_transform(*args, **kwargs)
        MLPClassifier.fit(self,Y,args[1], **kwargs)
        return self

    def predict(self, *args, **kwargs):
        Y = self.ch2.transform(*args, **kwargs)
        return MLPClassifier.predict(self,Y)
