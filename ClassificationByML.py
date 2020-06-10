import pandas as pd
import numpy as np
import xlrd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  classification_report
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import  MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn import svm
class SPRServer(object):
    def __init__(self):
        pass
    def _ReadExcel(self, path, y, strInfo):
        data = xlrd.open_workbook(path)
        data_table = data.sheet_by_name(data.sheet_names()[0])
        column = data_table.col_values(15)
        len_column =len(column)
        train_data_x_1 = []
        train_data_y_1 = [y for x in range(len_column)]
        for v in data_table.col_values(15):
            _list = v.split(",")
            data_list = [int(x) for x in _list]
            train_data_x_1.append(data_list)
        print(strInfo + str(len(train_data_x_1)) + " 标签长：" + str(len(train_data_y_1)))
        return train_data_x_1,train_data_y_1
    def _GetTrainAndTestDatas(self):
        path_train_1 = "D:\\datas\\075_R01_1.xlsx";
        train_data_x_1,train_data_y_1 = self._ReadExcel(path_train_1, 1, "训练集正样本数：")
        path_train_0 = "D:\\datas\\075_R01_0.xlsx";
        train_data_x_0, train_data_y_0 = self._ReadExcel(path_train_0, 0, "训练集负样本数：")
        train_data_x = train_data_x_1 + train_data_x_0
        train_data_y = train_data_y_1 + train_data_y_0
        path_test_1 = "D:\\datas\\test\\test_1.xlsx"
        test_data_x_1, test_data_y_1 = self._ReadExcel(path_test_1, 1, "测试集正样本数：")
        path_test_0 = "D:\\datas\\075_R01_test_0.xlsx"
        test_data_x_0, test_data_y_0 = self._ReadExcel(path_test_0, 0, "测试集负样本数：")
        test_data_x = test_data_x_1 + test_data_x_0
        test_data_y = test_data_y_1 + test_data_y_0
        return train_data_x,train_data_y,test_data_x,test_data_y
    def Machine_Learning(self):
        path_train_1 = "D:\\datas\\075_R01_1.xlsx";
        train_data_x_1,train_data_y_1 = self._ReadExcel(path_train_1, 1, "训练集正样本数：")
        path_train_0 = "D:\\datas\\075_R01_0.xlsx";
        train_data_x_0, train_data_y_0 = self._ReadExcel(path_train_0, 0, "训练集负样本数：")
        train_data_x = train_data_x_1 + train_data_x_0
        train_data_y = train_data_y_1 + train_data_y_0
        lr = LogisticRegression(max_iter=30000)
        lr.fit(train_data_x, train_data_y)
        sgdc = SGDClassifier()
        sgdc.fit(train_data_x, train_data_y)
        lsvc = LinearSVC(max_iter=300000000)
        lsvc.fit(train_data_x, train_data_y)
        mnb = MultinomialNB()
        mnb.fit(train_data_x, train_data_y)
        dtc = DecisionTreeClassifier()
        dtc.fit(train_data_x, train_data_y)
        rfc = RandomForestClassifier()
        rfc.fit(train_data_x, train_data_y)
        gbc = GradientBoostingClassifier()
        gbc.fit(train_data_x, train_data_y)
        with open("D:\\datas\\lr.pkl", "wb") as f:
            pickle.dump(lr, f)
        with open("D:\\datas\\sgdc.pkl", "wb") as f:
            pickle.dump(sgdc, f)
        with open("D:\\datas\\lsvc.pkl", "wb") as f:
            pickle.dump(lsvc, f)
        with open("D:\\datas\\mnb.pkl", "wb") as f:
            pickle.dump(mnb, f)
        path_test_1 = "D:\\datas\\test\\test_1.xlsx"
        test_data_x_1, test_data_y_1 = self._ReadExcel(path_test_1, 1, "测试集正样本数：")
        path_test_0 = "D:\\datas\\075_R01_test_0.xlsx"
        test_data_x_0, test_data_y_0 = self._ReadExcel(path_test_0, 0, "测试集负样本数：")
        test_data_x = test_data_x_1 + test_data_x_0
        test_data_y = test_data_y_1 + test_data_y_0
        lr_y_predict = lr.predict(test_data_x)
        sgdc_y_predict = sgdc.predict(test_data_x)
        lsvc_y_predict = lsvc.predict(test_data_x)
        mnb_y_predict = mnb.predict(test_data_x)
        dtc_y_predict = dtc.predict(test_data_x)
        rfc_y_predict = rfc.predict(test_data_x)
        gbc_y_predict = gbc.predict(test_data_x)
        print("LR分类器准确率：%s"%str(lr.score(test_data_x, test_data_y)))
        print(classification_report(test_data_y, lr_y_predict))
        print(lr.coef_)
        print(lr.intercept_)
        print("SGDC分类器准确率：%s" % str(sgdc.score(test_data_x, test_data_y)))
        print(classification_report(test_data_y, sgdc_y_predict))
        print("线性SVM分类器准确率：%s" % str(lsvc.score(test_data_x, test_data_y)))
        print(classification_report(test_data_y, lsvc_y_predict))
        print("朴素贝叶斯分类器准确率：%s" % str(mnb.score(test_data_x, test_data_y)))
        print(classification_report(test_data_y, mnb_y_predict))
        print("决策树分类器准确率：%s" % str(dtc.score(test_data_x, test_data_y)))
        print(classification_report(test_data_y, dtc_y_predict))
        print("随机森林分类器准确率：%s" % str(rfc.score(test_data_x, test_data_y)))
        print(classification_report(test_data_y, rfc_y_predict))
        print("梯度提升决策树准确率：%s" % str(gbc.score(test_data_x, test_data_y)))
        print(classification_report(test_data_y, gbc_y_predict))

    def LR_Machine_Learning(self):
        data = xlrd.open_workbook("D:\\datas\\test\\test_20200601.xlsx")
        data_table = data.sheet_by_name(data.sheet_names()[0])
        test_data_x = []
        for v in data_table.col_values(15):
            _list = v.split(",")
            print(_list)
            data_list = [int(x) for x in _list]
            test_data_x.append(data_list)
        with open("D:\\datas\\lsvc.pkl", "rb") as f:
            lsvc = pickle.load(f)
            print("线性SVM预测：" + str(lsvc.predict(test_data_x)))
        with open("D:\\datas\\lr.pkl", "rb") as f:
            lr = pickle.load(f)
            print("逻辑回归预测：" + str(lr.predict(test_data_x)))
    def LR_Curve(self):
        train_x, train_y, test_x, test_y = self._GetTrainAndTestDatas();
        lr = LogisticRegression(max_iter=30000)
        lr.fit(train_x, train_y)
        lr_y_predict = lr.predict(test_x)
        right_score = lr.score(test_x, test_y)
        ambu_metric = classification_report(test_y, lr_y_predict)
        y_pre = lr.predict_proba(test_x)
        y_0 = list(y_pre[:, 1])  # 取第二列数据，因为第二列概率为趋于0时分类类别为0，概率趋于1时分类类别为1
        fpr, tpr, thresholds = metrics.roc_curve(test_y, y_0)  # 计算fpr,tpr,thresholds
        auc = metrics.roc_auc_score(test_y, y_0)  # 计算auc
        print("LR算法准确率：" + str(right_score))
        print(ambu_metric)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('$ROC curve$')
        plt.show()
    def getCorr(self):
        train_x, train_y, test_x, test_y = self._GetTrainAndTestDatas();
        x = np.array(train_x)
        y = np.array(train_y)
        print(np.corrcoef(x, x))
    def getGraph(self):
        train_x, train_y, test_x, test_y = self._GetTrainAndTestDatas();
        lr = SGDClassifier()
        lr.fit(train_x, train_y)
        y_pre = lr.predict(test_x)
        fpr, tpr, thresholds = metrics.roc_curve(test_y, y_pre)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('$ROC curve$')
        plt.show()
if __name__ == "__main__":
    SPRServer().getGraph()




