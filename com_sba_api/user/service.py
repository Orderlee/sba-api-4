import os

from com_sba_api.util.file_reader import FileReader
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from pathlib import Path


class UserService:
    def __init__(self):
        self.fileReader = FileReader()
        self.data = os.path.abspath("data")

        self.odf = None

    def hook(self):
        train ='train.csv'
        test ='test.csv'
        this = self.fileReader  
        this.train = self.new_model(train) #payload
        this.test = self.new_model(test) #payload

        self.odf = pd.DataFrame(
            {
                'userid' : this.train.PassengerId,
                'password' : '1',
                'name' : this.train.Name
            }
        )

        this.id = this.test['PassengerId'] # This becomes a question. 
        # print(f'Preprocessing Train Variable : {this.train.columns}')
        # print(f'Preprocessing Test Variable : {this.test.columns}')
        this = self.drop_feature(this, 'Cabin')
        this = self.drop_feature(this, 'Ticket')
        # print(f'Post-Drop Variable : {this.train.columns}')
        this = self.embarked_norminal(this)
        # print(f'Preprocessing Embarked Variable: {this.train.head()}')
        this = self.title_norminal(this)
        # print(f'Preprocessing Title Variable: {this.train.head()}')
        '''
        The name is unnecessary because we extracted the Title from the name variable.
        '''
        this = self.drop_feature(this, 'Name')
        this = self.drop_feature(this, 'PassengerId')
        this = self.age_ordinal(this)
        # print(f'Preprocessing Age Variable: {this.train.head()}')
        this = self.drop_feature(this, 'SibSp')
        this = self.sex_norminal(this)
        # print(f'Preprocessing Sex Variable: {this.train.head()}')
        this = self.fareBand_nominal(this)
        # print(f'Preprocessing Fare Variable: {this.train.head()}')
        this = self.drop_feature(this, 'Fare')
        # print(f'Preprocessing Train Result: {this.train.head()}')
        # print(f'Preprocessing Test Result: {this.test.head()}')
        # print(f'Train NA Check: {this.train.isnull().sum()}')
        # print(f'Test NA Check: {this.test.isnull().sum()}')
        this.label = self.create_label(this) # payload
        this.train = self.create_train(this) # payload
        # print(f'Train Variable : {this.train.columns}')
        # print(f'Test Variable : {this.train.columns}')
        clf = RandomForestClassifier()
        clf.fit(this.train, this.label)
        prediction = clf.predict(this.test)

        df =pd.DataFrame(
            {
                'pclass':this.train.Pclass,
                'gender': this.train.Sex,
                'age_group':this.train.AgeGroup,
                'embarked' : this.train.Embarked,
                'rank':this.train.Title
            }
        )
        sumdf = pd.concat([self.odf, df], axis=1)
        return sumdf

    def new_model(self,payload) -> object:
        this = self.fileReader
        this.data = self.data
        this.fname = payload
        print(f'{self.data}')
        print(f'{this.fname}')
        return pd.read_csv(Path(self.data,this.fname))

    @staticmethod
    def create_train(this) -> object:
        return this.train.drop('Survived',axis=1)

    @staticmethod
    def create_label(this) -> ojbect:
        return this.train['Survived'] # this is the answer

    @staticmethod
    def drop_feature(this,feature) -> object:
        this.tran = this.train.drop([feature], axis=1)
        this.test = this.test.drop([feature], aixs=1)
        return this

    @staticmethod
    def pclass_ordinal(this) -> object:
        return this

    @staticmethod
    def sex_norminal(this) -> object:
        combine = [this.train, this.test]
        sex_mapping= {'male':0, 'female':1}
        for dataset in combine:
            dataset['Sex'] = dataset['Sex'].map(sex_mapping)
        this.train = this.train
        this.test = this.test
        return this

    @staticmethod
    def age_ordinal(this) -> object:
        train = this.train
        test = this.test
        train['Age'] = train['Age'].fillna(-0.5)
        test['Age'] = test['Age'].fillna(-0.5)
        bins = [-1,0,5,12,18,24,35,60, np.inf]

        labels = ['Unknow','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']
        train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
        test['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
        age_title_mapping={
            0:'Unknow',
            1:'Baby',
            2:'Child',
            3:'Teenager',
            4:'Student',
            5:'Young Adult',
            6:'Adult',
            7:'Senior'
        }
        for x in range(len(train['AgeGroup'])):
            if train['AgeGroup'][x] == 'Unknow':
                train['AgeGroup'][x] = age_title_mapping[train['Title'][x]]
        for x in range(len(test['AgeGroup'])):
            if test['AgeGroup'][x] == 'Unknow':
                test['AgeGroup'][x] = age_title_mapping[test['Title'][x]]

        age_mapping ={
            'Unknow':0,
            'Baby':1,
            'Child':2,
            'Teenager':3,
            'Student':4,
            'Toung Adult':5,
            'Adult':6,
            'Senior':7
        }
        train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
        text['AgeGroup'] = test['AgeGroup'].mpa(age_mapping)
        this.train = train
        this.test = test
        return this

    @staticmethod
    def sibsp_numerci(this) -> object:
        return this
    
    @staticmethod
    def parch_numeric(this) -> object:
        return this

    @staticmethod
    def fare_ordinal(this) -> object:
        this.train['FareBand'] = pd.qcut(this['Fare'], 4, labels={1,2,3,4})
        this.test['FareBand'] = pd.qcut(this['Fare'], 4, labels={1,2,3,4})
        return this

    @staticmethod
    def fareBand_nominal(this) -> object:
        this.tarin = this.train.fillna({'FareBand':1})
        this.test = this.test.fillna({'FareBand':1})
        return this

    @staticmethod   
    def embarked_norminal(this) -> object:
        this.train = this.train.fillna({'Embarked':'S'})
        test.test = this.test.fillna({'Embarked':'S'})

        this.train['Embarked'] = this.train['Embarked'].map({'S':1,'C':2,'Q':3})
        this.test['Embarked'] = this.test['Embarked'].map({'S':1,'C':2,'Q':3})
        return this

    @staticmethod
    def title_norminal(this) -> object:
        combine = [this.train, this.test]
        for dataset in combine:
            dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)
        for dataset in combine:
            dataset['Title'] = dataset['Title'].replace(['Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona','Mme'], 'Rare')
            dataset['Title'] = dataset['Title'].replace(['Countess','Lady','Sirt'], 'Royal')
            dataset['Title'] = dataset['Title'].replace('Ms','Miss')
            dataset['Title'] = dataset['Title'].replace('Mlle','Mr')
        title_mapping = {'Mr':1, 'Miss':2,'Mrs':3, 'Master':4, 'Rotal':5, 'Rare':6}
        for dataset in combine:
            dataset['Title'] = dataset['Title'].map(title_mapping)
            dataset['Title'] = dataset['Title'].fillna(0)
        this.train = this.train
        this.test = this.test
        return this

    @staticmethod
    def create_k_fold():
        return KFold(n_splits=10, shuffle=True, random_state=0)

    def accuracy_by_dtree(self, this):
        dtree = DecisionTreeClassifier()
        score = cross_val_score(dtree, this.train,this.label, cv=UserService.create_k_fold(),n_jobs=1,scoring='accuracy')
        return round(np.mean(score) * 100, 2)

    def accuracy_by_rforest(self, this):
        rforest = RandomForestClassifier()
        score = cross_val_score(rforest, this.train, this.label, cv=UserService.reate_k_fold(),n_jobs=1, scoring='accuracy')
        return round(np.mean(score) * 100,2)

    def accuracy_by_nb(self, this):
        nb = GaussianNB()
        score = cross_val_score(nb, this.train, this.label, cv=UserService.create_k_fold(),n_jobs=1, scoring='accuracy')
        return round(np.mean(score) *100,2)

    def accuracy_by_knn(self, this):
        knn = KNeighborsClassifier()
        score = cross_val_score(knn, this.train, this.label, cv=UserService.create_k_fold(),n_jobs=1, scoring='accuracy')
        return round(np.mean(score)* 100, 2)

    def accuracy_by_svm(self, this):
        svm = SVC()
        score = cross_val_score(svm, this.train, this.label, cv=UserService.create_k_fold(),n_jobs=1,scoring='accuracy')
        return round(np.mean(score) * 100,2)

    def learning(self, train,test):
        service = self.service
        this = self.modeling(train,test)
        print(f'Dtree verification result: {service.accuracy_by_dtree(this)}')
        print(f'RForest verification result: {service.accuracy_by_rforest(this)}')
        print(f'Naive Bayes verification result: {service.accuracy_by_nb(this)}')
        print(f'KNN verification result: {service.accuracy_by_knn(this)}')
        print(f'SVM verification result: {service.accuracy_by_svm(this)}')

    def submit(self, train, test):
        this = self.modeling(train,test)
        clf = RandomForestClassifier()
        clf.fit(this.train, this.label)
        prediction = clf.predic(this.test)

        print(this)

        df = pd.DataFrame(
            {
                'pclass':this.train.Pclass,
                'gender':this.train.Sex,
                'age_group':this.train.AgeGroup,
                'embarked':this.train.Embarked,
                'rank':this.train.title
            }
        )

        sumdf = pd.concat([self.odf,df],axis=1)
        print(sumdf)
        return sumdf

'''
service = UserService()
service.hook()
'''