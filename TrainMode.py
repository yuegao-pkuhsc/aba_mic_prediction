# encoding: utf-8
"""
Created on Thu Apr 20 17:33:47 2023

@author:YueGao
"""

import re
import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from joblib import load
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
import logging
import datetime
import sys
from sklearn.model_selection import StratifiedShuffleSplit

class  AbaumanniiTrainModel:
    def loginit(outPath):
        """
        Output of logs, including console and files
        """
        AbaumanniiTrainModel.logger = logging.getLogger('Yue.Gao')
        AbaumanniiTrainModel.logger.setLevel(logging.INFO)
        rf_handler = logging.StreamHandler(sys.stdout)  # default: sys.stderr
        rf_handler.setLevel(logging.INFO)
        rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))

        f_handler = logging.FileHandler(r"" + outPath + 'out.log')
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))

        AbaumanniiTrainModel.logger.addHandler(rf_handler)
        AbaumanniiTrainModel.logger.addHandler(f_handler)

    def __init__(self,drug_name,sourcePath,sourceFile,phenoFile,modelPath,outPath,worktimes=1):
       """
        Initialize the environment of machine learning
       :param drug_name:Name of the antibiotic used for prediction
       :param sourcePath:Path to the source file
       :param sourceFile:Name of the k-mer file
       :param phenoFile:Name of the MIC data file
       :param modelPath:Path to the model file
       :param outPath:Output path
       """
       self.drug_name=drug_name
       self.sourcePath=sourcePath
       self.sourceFile=sourceFile
       self.phenoFile=phenoFile
       self.modelPath=modelPath
       self.outPath=outPath
       self.__loadSourceFile()
       self.__readPheno()
       self.__makeData()
       self.work_times=worktimes
       self.acc = {
           "RandomForest": 0,
           "SVMLinear": 0,
           "SVMPloy": 0,
           "SVMRbf": 0,
           "Xgboost": 0
       }
       self.loadRes = {
           "RandomForest":[0,"","","",""] ,
           "SVMLinear": [0,"","","",""],
           "SVMPloy": [0,"","","",""],
           "SVMRbf": [0,"","","",""],
           "Xgboost":[0,"","","",""]
       }
       self.isFeature=True
       self.iTrainCount=5
       plt.figure()

    def __loadSourceFile(self):
        """
        Read in the k-mer file and generate dataframe, format required CSV file
        :return:

        """
        self.sourceData = pd.read_csv(r"" + self.sourcePath  + self.sourceFile, header=0, index_col=0, low_memory=False)
        AbaumanniiTrainModel.logger.info(self.sourceData.head())
        try:
            self.sourceData = self.sourceData.drop('Sample', axis=0)
        except:
            AbaumanniiTrainModel.logger.info("No line Sample!!")
        da_array = self.sourceData.values
        AbaumanniiTrainModel.logger.info(da_array.shape)
        self.dataID = list(self.sourceData.index.values)

    def __countROCAUC(self,y_test, y_score, drug_name, clfname, n_classes):
            """
            Calculate the values of ROC and AUC
            :param y_test:
            :param y_score:
            :param drug_name:
            :param clfname:
            :param n_classes:
            :return:
            """
            y_test = np.array(y_test)
            AbaumanniiTrainModel.logger.info(y_test.shape)
            y_score = np.array(y_score)
            AbaumanniiTrainModel.logger.info(y_score.shape)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(self.n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            AbaumanniiTrainModel.logger.info(fpr)
            AbaumanniiTrainModel.logger.info(tpr)
            AbaumanniiTrainModel.logger.info(roc_auc)

            return fpr, tpr, roc_auc

    def __readPheno(self):
           """
            Read in the MIC data file and normalize the data
           """
           # extract the ID number of the strains
           self.ID = []
           self.MIC = []
           phenoData=""
           with open(r"" + self.sourcePath + self.phenoFile) as myData:
               phenoData = myData.readlines()
           for i in range(len(self.dataID)):
                for line in phenoData:
                    x=re.split(r"[ \t]+", line)
                    tmp1 = x[1].strip('\n')
                    if tmp1 == 'NA':
                        continue
                    if x[0] ==self.dataID[i]:
                        self.ID.append(x[0])
                        self.MIC.append(tmp1)

           AbaumanniiTrainModel.logger.info("Pheno ID is {}".format(self.ID))

    def  __makeRandomForest(self,X_train,y_train,X_test,y_test):
        n_estimators_tmp = 600
        model = OneVsRestClassifier(RandomForestClassifier(n_estimators=n_estimators_tmp, n_jobs=-1))
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        if acc > self.acc["RandomForest"]:
            self.acc["RandomForest"] = acc
            AbaumanniiTrainModel.logger.info('RandomForestClassifier make  accuracy:{}'.format(acc))
            joblib.dump(model, r"" + self.modelPath + self.drug_name + '_RF_clf.pkl')


    def  __makeSVMLinear(self,X_train,y_train,X_test,y_test):
        clflinear = OneVsRestClassifier(svm.SVC(kernel='linear', C=1, probability=True)).fit(X_train, y_train)
        acc = clflinear.score(X_test, y_test)
        if acc > self.acc["SVMLinear"]:
            self.acc["SVMLinear"] = acc
            AbaumanniiTrainModel.logger.info('SVM-linear  make accuracy: {}'.format(acc))
            joblib.dump(clflinear, r"" + self.modelPath + self.drug_name + '_SVMlinear_clf.pkl')


    def  __makeSVMPoly(self,X_train,y_train,X_test,y_test):
        clfpoly = OneVsRestClassifier(svm.SVC(kernel="poly", degree=5, coef0=1, C=1, probability=True)).fit(X_train,
                                                                                                            y_train)  # (gamma*u'*v + coef0)^degree
        acc = clfpoly.score(X_test, y_test)
        if acc > self.acc["SVMPloy"]:
            self.acc["SVMPloy"] = acc
            AbaumanniiTrainModel.logger.info('SVM-poly  make accuracy:{} '.format(acc))
            joblib.dump(clfpoly,  r"" + self.modelPath + self.drug_name + '_SVMpoly_clf.pkl')


    def  __makeSVMRbf(self,X_train,y_train,X_test,y_test):
        clfrbf = OneVsRestClassifier(svm.SVC(kernel="rbf", degree=5, coef0=1, C=1, probability=True)).fit(X_train,
                                                                                                          y_train)  # sigmoid：tanh(gamma*u'*v + coef0)

        acc = clfrbf.score(X_test, y_test)
        if acc > self.acc["SVMRbf"]:
            self.acc["SVMRbf"] = acc
            AbaumanniiTrainModel.logger.info('SVM-rbf  make accuracy:{} '.format(acc))
            joblib.dump(clfrbf,  r"" + self.modelPath + self.drug_name + '_SVMrbf_clf.pkl')

    def  __makeXgboost(self,X_train,y_train,X_test,y_test):
        model = OneVsRestClassifier(xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=50,
                                                      silent=True, objective='binary:logistic', n_jobs=-1))

        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)
        acc = accuracy_score(y_test, y_preds)
        if acc > self.acc["Xgboost"]:
            self.acc["Xgboost"] = acc
            AbaumanniiTrainModel.logger.info("xgboost make accuracy :{}".format(acc))
            joblib.dump(model,  r"" + self.modelPath + self.drug_name + '_XGBoost_clf.pkl')

    def makeTrainModel(self):
        Y=self.Y_MIC
        self.n_classes = len(np.unique(Y))
        AbaumanniiTrainModel.logger.info(self.n_classes)
        # LabelBinarizer - one hot
        lb = LabelBinarizer()
        Y = lb.fit_transform(Y)

        # self.ID
        AbaumanniiTrainModel.logger.info(self.drug_name)
        X=self.X
        X_train,X_test,y_train,y_test=([],[],[],[])
        shuff = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
        AbaumanniiTrainModel.logger.info(X.shape)
        AbaumanniiTrainModel.logger.info(Y.shape)
        iTrainCount = 0
        for train_index, test_index in shuff.split(X, Y):
            self.tag = 0
            iTrainCount+=1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            AbaumanniiTrainModel.logger.info('train strains: \n')

            __outTrainState = ""
            for i in range(len(train_index)):
                __outTrainState += self.ID[train_index[i]] + ','
            AbaumanniiTrainModel.logger.info(__outTrainState)
            AbaumanniiTrainModel.logger.info('\n')
            AbaumanniiTrainModel.logger.info('test strains: \n')

            __outTestState = ""
            for i in range(len(test_index)):
                __outTestState += self.ID[test_index[i]] + ','
            AbaumanniiTrainModel.logger.info(__outTestState)
            y_test_true = np.argmax(y_test, axis=1)
            AbaumanniiTrainModel.logger.info('test_label:{} '.format(y_test_true))
            AbaumanniiTrainModel.logger.info('\n')
            if (self.isFeature and iTrainCount>=self.iTrainCount):
                break
            self.__makeRandomForest(X_train, y_train, X_test, y_test)
            self.__makeSVMLinear( X_train, y_train, X_test, y_test)
            self.__makeSVMPoly( X_train, y_train, X_test, y_test)
            self.__makeSVMRbf( X_train, y_train, X_test, y_test)
            self.__makeXgboost(X_train, y_train, X_test, y_test)


    def loadRandomForestModel(self,eigenvalue=False):
        RFmodel=""
        for i in range(10):
            RFmodel = load(r"" + self.modelPath + self.drug_name + '_RF_clf.pkl')
        algorithms="RandomForest"
        color_plot="deeppink"
        self.__loadModel(RFmodel,algorithms,color_plot,eigenvalue,self.work_times)

    def loadSVMLinearTrainModel(self,eigenvalue=False):
        clflinear = load(r"" + self.modelPath + self.drug_name + '_SVMlinear_clf.pkl')
        algorithms = "SVMLinear"
        color_plot = "navy"
        self.__loadModel(clflinear, algorithms, color_plot, eigenvalue,self.work_times)

    def loadSVMPloyTrainModel(self,eigenvalue=False):
        clfpoly = load(r"" + self.modelPath + self.drug_name + '_SVMpoly_clf.pkl')
        algorithms = "SVMPloy"
        color_plot = "aqua"
        self.__loadModel(clfpoly, algorithms, color_plot, eigenvalue, self.work_times)

    def loadSVMRbfTrainModel(self,eigenvalue=False):
        clfrbf = load(r"" + self.modelPath + self.drug_name + '_SVMrbf_clf.pkl')
        algorithms = "SVMRbf"
        color_plot = "darkorange"
        self.__loadModel(clfrbf, algorithms, color_plot, eigenvalue, self.work_times)

    def loadXgboostTrainModel(self,eigenvalue=False):
        XGBmodel = load(r"" + self.modelPath + self.drug_name + '_XGBoost_clf.pkl')
        algorithms = "Xgboost"
        color_plot = "cornflowerblue"
        self.__loadModel(XGBmodel, algorithms, color_plot, eigenvalue, self.work_times)

    def __loadModel(self,RFmodel,algorithms,color_plot,eigenvalue,work_times):
        Y = self.Y_MIC
        self.n_classes = len(np.unique(Y))
        AbaumanniiTrainModel.logger.info(self.drug_name + ":{}".format(self.n_classes))
        # LabelBinarizer - one hot
        lb = LabelBinarizer()
        Y = lb.fit_transform(Y)
        #RFmodel.
        # self.ID
        AbaumanniiTrainModel.logger.info(self.drug_name)
        X = self.X
        localSeq = 0
        X_train, X_test, y_train, y_test = ([], [], [], [])
        for i in range(work_times):
            shuff = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
            AbaumanniiTrainModel.logger.info(X.shape)
            AbaumanniiTrainModel.logger.info(Y.shape)

            iTrainCount = 0
            for train_index, test_index in shuff.split(X, Y):
                localSeq+=1
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]

                y_RF_score = RFmodel.predict_proba(X_test)
                AbaumanniiTrainModel.logger.info(y_RF_score)
                RF_fpr, RF_tpr,RF_roc_auc = self.__countROCAUC(y_test, y_RF_score, self.drug_name,
                                                               algorithms, self.n_classes)
                iTrainCount+=1
                if (self.isFeature and iTrainCount >= self.iTrainCount):
                    break
                acc = RFmodel.score(X_test, y_test)
                if acc > self.loadRes[algorithms][0]:
                    self.loadRes[algorithms][0] = acc
                    self.loadRes[algorithms][1]= RF_fpr
                    self.loadRes[algorithms][2] =RF_tpr
                    self.loadRes[algorithms][3] =RF_roc_auc
                y_pred = RFmodel.predict(X_test)
                y_test_true = np.argmax(y_test, axis=1)
                y_pred_true = np.argmax(y_pred, axis=1)
                AbaumanniiTrainModel.logger.info('{} {} predit_label: {}'.format(self.drug_name, algorithms, y_pred_true))
                AbaumanniiTrainModel.logger.info('\n')
                AbaumanniiTrainModel.logger.info('{} {}Classifier{}: {}'.format(self.drug_name, algorithms, localSeq, acc))
                randomForestConfusion = confusion_matrix(y_test_true, y_pred_true)
                AbaumanniiTrainModel.logger.info("{} {} confusion_matrix{}：{}".format(self.drug_name,algorithms,localSeq,randomForestConfusion))
                AbaumanniiTrainModel.logger.info('{} {} micro{}:{}'.format(self.drug_name,algorithms,localSeq,RF_roc_auc["micro"]))
                np.savetxt(self.outPath + "confusion_"+ algorithms +  str(localSeq)  + "_load.csv",np.array(randomForestConfusion), delimiter=",",fmt='%d')


            plt.plot(self.loadRes[algorithms][1]["micro"],self.loadRes[algorithms][2]["micro"],
                     label=' ' + algorithms + ' ROC curve (area = {0:0.2f})'
                           ''.format( self.loadRes[algorithms][3]['micro'] ),
                     color=color_plot, linewidth=2)

            if eigenvalue and algorithms!= "SVMLinear":
                importances = pd.DataFrame(
                    {'feature': self.sourceData.columns, 'importance': np.round(RFmodel.estimators_[-1].feature_importances_, 3)})
                importances = importances.sort_values('importance', ascending=False).set_index('feature')
                importances=importances[:500]
                col=["feature","importance"]
                testcsv = pd.DataFrame(columns=col, data=importances)
                testcsv.to_csv(r"" + self.outPath + "import_" +  algorithms + ".csv")
                AbaumanniiTrainModel.logger.info(importances)

            if eigenvalue and algorithms== "SVMLinear":
                svm_weights_1 = (RFmodel.estimators_[-1].coef_ ** 2).ravel()
                svm_weights_2 = svm_weights_1 / svm_weights_1.sum()
                importances = pd.DataFrame(
                    {'feature': self.sourceData.columns, 'importance': np.round(svm_weights_2, 3)})
                importances = importances.sort_values('importance', ascending=False).set_index('feature')
                importances = importances[:500]
                col = ["feature", "importance"]
                testcsv = pd.DataFrame(columns=col, data=importances)
                testcsv.to_csv(r"" + self.outPath + "import_"  + algorithms +  "learn.csv")
                AbaumanniiTrainModel.logger.info(importances)


    def outROCAUC(self):
        # print picture
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 5,
                 }
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating curve')
        plt.legend(loc="lower right", prop=font1)
        plt.savefig(r"" + self.outPath + self.drug_name + '-ROC-AUC-all.pdf')

    def __makeData(self):
         AbaumanniiTrainModel.logger.info(self.ID)
         AbaumanniiTrainModel.logger.info(self.MIC)

         value_cnt = {}  # Store the result in a dictionary
         for value in self.MIC:
             value_cnt[value] = value_cnt.get(value, 0) + 1
         # Print out the results
         AbaumanniiTrainModel.logger.info(value_cnt)
         AbaumanniiTrainModel.logger.info([key for key in value_cnt.keys()])
         AbaumanniiTrainModel.logger.info([value for value in value_cnt.values()])

         self.Y_MIC = []
         DIC_MIC={  '256':15,
                    '128':14,
                    '64':13,
                    '32':12,
                    '16':11,
                    '8':10,
                    '4':9,
                    '2':8,
                    '1':7,
                    '0.5':6,
                    '0.25':5,
                    '0.125':4,
                    '0.064':3,
                    '0.032':2,
                    '0.016':1}
         for value in self.MIC:
             if value  in DIC_MIC.keys():
                 self.Y_MIC.append(DIC_MIC[value])

         # Display the statistical results of Y_test
         value_cnt = {}
         for value in self.Y_MIC:
             value_cnt[value] = value_cnt.get(value, 0) + 1
         AbaumanniiTrainModel.logger.info(value_cnt)
         AbaumanniiTrainModel.logger.info([key for key in value_cnt.keys()])
         AbaumanniiTrainModel.logger.info([value for value in value_cnt.values()])

         self.Y_MIC = np.array(self.Y_MIC)
         AbaumanniiTrainModel.logger.info(self.Y_MIC.shape)

         X_da = self.sourceData.loc[self.sourceData.index.intersection(self.ID)]
         self.X = X_da.values
         AbaumanniiTrainModel.logger.info(self.X.shape)

work_file={

"AMK":["AMK_featurecol.csv","aba_mic_AMK.txt"],
"CAZ":["CAZ_featurecol.csv","aba_mic_CAZ_minus1.txt"],
"CIP":["CIP_featurecol.csv","aba_mic_CIP_minus1.txt"],
"COL":["COL_featurecol.csv","aba_mic_COL_minus1.txt"],
"CSL":["CSL_featurecol.csv","aba_mic_CSL_minus2.txt"],
"FEP":["FEP_featurecol.csv","aba_mic_FEP_minus2.txt"],
"IPM":["IPM_featurecol.csv","aba_mic_IPM.txt"],
"LVX":["LVX_featurecol.csv","aba_mic_LVX_minus1.txt"],
"MEM":["MEM_featurecol.csv","aba_mic_MEM_minus2.txt"],
"MNO":["MNO_featurecol.csv","aba_mic_MNO_minus1.txt"],
"SXT":["SXT_featurecol.csv","aba_mic_SXT_48.txt"],
"TGC":["TGC_featurecol.csv","aba_mic_TGC.txt"],
"TZP":["TZP_featurecol.csv","aba_mic_TZP_minus1.txt"]

}
def outAll():
    AbaumanniiTrainModel.loginit("./out/")
    for key in work_file:
        r = AbaumanniiTrainModel(key, "./source/", work_file[key][0], work_file[key][1], "./model/" + key + "/",
                                 "./out/" + key + "/")
        r.makeTrainModel()
        r.loadRandomForestModel()
        r.loadSVMLinearTrainModel()
        r.loadSVMPloyTrainModel()
        r.loadSVMRbfTrainModel()
        r.loadXgboostTrainModel()
        r.outROCAUC()


def outAllQue():
    AbaumanniiTrainModel.loginit("./out/")
    for key in work_file:
        r = AbaumanniiTrainModel(key, "./source/", work_file[key][0], work_file[key][1], "./model/" + key + "/",
                                 "./out/" + key + "/",1)

        r.makeTrainModel()
        r.loadRandomForestModel()
        r.loadRandomForestModel()
        r.outROCAUC()


def outOneAMK():
        AbaumanniiTrainModel.loginit("./out/")
        r = AbaumanniiTrainModel("your-drug-name", "./source/", "k11_v1.csv","aba_mic_TGC_minus2.txt" , "./model_sever/" + "TGC" + "/",
                      "./out/" +"TGC" + "/",2)
        r.loadRandomForestModel()
if __name__ == "__main__":
       # outOneAMK()
        outAll()