
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy  
from sklearn.model_selection import validation_curve
import time
get_ipython().run_line_magic('matplotlib', 'inline')
path='/SOT/'

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn import tree



    for model in model_list:
        clf = model(parameter)
        clf = clf.fit(x_train,y_train)
        print(model.__name__,clf.score(x_test,y_test))


## original dataset

#original dataset
print('start training for orginal dataset')
ori_df=pd.read_csv("train_1000.csv")
ori_df.head()
print(ori_df.shape)
y=np.array(ori_df['Sentiment'])
data=np.array(ori_df)
x=data[:,2:]
print(x.shape)
print(y.shape)
y_train=y[:800]
x_train=x[:800,:]
y_test=y[800:]
x_test=x[800:,:]
print(x_train.shape)





# # SVM

from sklearn import svm
svc = svm.SVC()
param_range = np.logspace(-4, 2, 6)
train_scores, test_scores = validation_curve(
    svc, x_train,y_train, param_name="gamma", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=1)
#calculate mean and std for plot
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.figure(figsize=(8,6)) 
plt.title("SVM Validation Curve with for gamma")
plt.style.use('seaborn-poster')
plt.xlabel("gamma")
plt.ylabel("Score")
plt.ylim(round(min(min(test_scores_mean),min(train_scores_mean)),1)-0.1, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,lw=lw)
plt.semilogx(param_range, test_scores_mean, label="5-fold cross-validation score",lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2 ,lw=lw)
plt.legend(loc="best")
plt.savefig('Step5.1_svm_ori_gamma_ValiCurve.png')
plt.show()
optimum_gamma = param_range[list(test_scores_mean).index(max(test_scores_mean))]
print('Plot saved')
print('The maximum validation accuracy is:', max(test_scores_mean),'\n'
      'The corresponding traning accuracy is:', train_scores_mean[list(test_scores_mean).index(max(test_scores_mean))],'\n'
      'The optimum gamma is:',optimum_gamma)
svc = svm.SVC(gamma = optimum_gamma )
svc = svc.fit(x_train, y_train)
print('The test accuracy is:',svc.score(x_test,y_test))

# C
#set the optimum gamma
param_range = np.linspace(0.01, 6, 10)
train_scores, test_scores = validation_curve(
    svc, x_train,y_train, param_name="C", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=1)
#calculate mean and std for plot
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
#plot the tranning results
plt.figure(figsize=(8,6)) 
plt.title("SVM Validation Curve with for C")
plt.style.use('seaborn-poster')
plt.xlabel("C")
plt.ylabel("Score")
plt.ylim(round(min(min(test_scores_mean),min(train_scores_mean)),1)-0.1, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, lw=lw)
plt.plot(param_range, test_scores_mean, label="5-fold cross-validation score", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, lw=lw)
plt.legend(loc="best")
plt.savefig('Step5.1_svm_ori_C_ValiCurve.png')
plt.show()
#print the optimum parameter and accuracy
optimum_c = (param_range[list(test_scores_mean).index(max(test_scores_mean))])
print('Plot saved')
print('The maximum validation accuracy is:', max(test_scores_mean),'\n'
      'The corresponding traning accuracy is:', train_scores_mean[list(test_scores_mean).index(max(test_scores_mean))],'\n'
      'The optimum C is:',optimum_c)
svc = svm.SVC(C = optimum_c, gamma = optimum_gamma)
svc = svc.fit(x_train, y_train)
#print the test score
print('The test accuracy is:',svc.score(x_test,y_test))


# # BernoulliNB

from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB()
clf = clf.fit(x_train,y_train)
param_range = np.linspace(0.001, 2, 50)
train_scores, test_scores = validation_curve(
    clf, x_train,y_train, param_name="alpha", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=1)
#calculate mean and std for plot
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.figure(figsize=(8,6)) 
plt.title("Bernoulli naive_bayes Validation Curve with for Alpha")
plt.xlabel("Alpha")
plt.ylabel("Score")
plt.style.use('seaborn-poster')
plt.ylim(round(min(min(test_scores_mean),min(train_scores_mean)),1)-0.1, 1.1)
lw = 2

plt.plot(param_range, train_scores_mean, label="Training score", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, lw=lw)
plt.plot(param_range, test_scores_mean, label="5-fold cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig('Step5.1_bayesan_ori_alpha_ValiCurve.png')
plt.show()
optimum_alpha = param_range[list(test_scores_mean).index(max(test_scores_mean))]
print('Plot saved')
print('The maximum validation accuracy is:', max(test_scores_mean),'\n'
      'The corresponding traning accuracy is:', train_scores_mean[list(test_scores_mean).index(max(test_scores_mean))],'\n'
      'The optimum alpha is:',optimum_alpha)
clf = BernoulliNB(alpha = optimum_alpha )
clf = clf.fit(x_train, y_train)

print('The test accuracy is:',clf.score(x_test,y_test))


# # LogisticRegression

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf = clf.fit(x_train,y_train)
param_range = np.logspace(-100, -50, 50)
train_scores, test_scores = validation_curve(
    clf, x_train,y_train, param_name="tol", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=1)
#calculate mean and std for plot
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.figure(figsize=(8,6)) 
plt.title("Logistic Regression Validation Curve with for tol")
plt.xlabel("tol")
plt.ylabel("Score")
plt.style.use('seaborn-poster')
plt.ylim(round(min(min(test_scores_mean),min(train_scores_mean)),1)-0.1, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, lw=lw)
plt.semilogx(param_range, test_scores_mean, label="5-fold cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig('Step5.1_Logistic Regression_tol_ValiCurve.png')
plt.show()
optimum_alpha = param_range[list(test_scores_mean).index(max(test_scores_mean))]
print('Plot saved')
print('The maximum validation accuracy is:', max(test_scores_mean),'\n'
      'The corresponding traning accuracy is:', train_scores_mean[list(test_scores_mean).index(max(test_scores_mean))],'\n'
      'The optimum alpha is:',optimum_alpha)
clf = LogisticRegression(tol = optimum_alpha )
clf = clf.fit(x_train, y_train)
print('The test accuracy is:',clf.score(x_test,y_test))

# C
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(tol = 0.0000000000000000001)
clf = clf.fit(x_train,y_train)
param_range = np.linspace(0.01, 2, 20)
train_scores, test_scores = validation_curve(
    clf, x_train,y_train, param_name="C", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=1)
#calculate mean and std for plot
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.figure(figsize=(8,6)) 
plt.title("Logistic Regression Validation Curve with for C")
plt.xlabel("C")
plt.ylabel("Score")
plt.style.use('seaborn-poster')
plt.ylim(round(min(min(test_scores_mean),min(train_scores_mean)),1)-0.1, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, lw=lw)
plt.plot(param_range, test_scores_mean, label="5-fold cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig('Step5.1_Logistic Regression_C_ValiCurve.png')
plt.show()
optimum_alpha = param_range[list(test_scores_mean).index(max(test_scores_mean))]
print('Plot saved')
print('The maximum validation accuracy is:', max(test_scores_mean),'\n'
      'The corresponding traning accuracy is:', train_scores_mean[list(test_scores_mean).index(max(test_scores_mean))],'\n'
      'The optimum alpha is:',optimum_alpha)
clf = LogisticRegression(C = optimum_alpha )
clf = clf.fit(x_train, y_train)

print('The test accuracy is:',clf.score(x_test,y_test))


# # LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis(solver="lsqr",shrinkage = 'auto')
clf = clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))



# # Perceptron

from sklearn.linear_model import Perceptron
clf = Perceptron()
clf = clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))

clf = Perceptron(penalty='l2')
clf = clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
# tol
clf = Perceptron()
clf = clf.fit(x_train,y_train)
param_range = np.logspace(-50, -10, 20)
train_scores, test_scores = validation_curve(
    clf, x_train,y_train, param_name="tol", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=1)
#calculate mean and std for plot
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8,6)) 
plt.title("Perceptron Validation Curve with for tol")
plt.xlabel("tol")
plt.ylabel("Score")
plt.style.use('seaborn-poster')
plt.ylim(round(min(min(test_scores_mean),min(train_scores_mean)),1)-0.1, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, lw=lw)
plt.semilogx(param_range, test_scores_mean, label="5-fold cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig('Step5.1_Perceptron_ValiCurve.png')
plt.show()
optimum_tol = param_range[list(test_scores_mean).index(max(test_scores_mean))]
print('Plot saved')
print('The maximum validation accuracy is:', max(test_scores_mean),'\n'
      'The corresponding traning accuracy is:', train_scores_mean[list(test_scores_mean).index(max(test_scores_mean))],'\n'
      'The optimum tol is:',optimum_tol)
clf = Perceptron(tol = optimum_tol )
clf = clf.fit(x_train, y_train)

print('The test accuracy is:',clf.score(x_test,y_test))


# # KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf = clf.fit(x_train,y_train)
param_range = range(1,5)
train_scores, test_scores = validation_curve(
    clf, x_train,y_train, param_name="n_neighbors", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=1)
#calculate mean and std for plot
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.figure(figsize=(8,6)) 
plt.title("K-Neighbors Classifier Curve with for n_neighbors")
plt.xlabel("n_neighbors")
plt.ylabel("Score")
plt.style.use('seaborn-poster')
plt.ylim(round(min(min(test_scores_mean),min(train_scores_mean)),1)-0.1, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, lw=lw)
plt.plot(param_range, test_scores_mean, label="5-fold cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig('Step5.1_KNeighborsClassifier_n_ValiCurve.png')
plt.show()
optimum_alpha = param_range[list(test_scores_mean).index(max(test_scores_mean))]
print('Plot saved')
print('The maximum validation accuracy is:', max(test_scores_mean),'\n'
      'The corresponding traning accuracy is:', train_scores_mean[list(test_scores_mean).index(max(test_scores_mean))],'\n'
      'The optimum alpha is:',optimum_alpha)
clf = KNeighborsClassifier(n_neighbors = optimum_alpha )
clf = clf.fit(x_train, y_train)
print('The test accuracy is:',clf.score(x_test,y_test))

# # PCA dataset

print('start training for PCA result')
df_DR=pd.read_csv('train_1000_DR.csv')
np_dr=np.array(df_DR)
x=np_dr[:,1:-1]
y=np_dr[:,1]
y_train=y[:800]
x_train=x[:800,:]
y_test=y[800:]
x_test=x[800:,:]
print(x_train.shape)
print(y.sum())

#KNeighborsClassifier

clf = KNeighborsClassifier()
clf = clf.fit(x_train,y_train)
param_range = range(1,10)
train_scores, test_scores = validation_curve(
    clf, x_train,y_train, param_name="n_neighbors", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=1)
#calculate mean and std for plot
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.figure(figsize=(8,6)) 
plt.title("K-Neighbors Classifier Curve with for n_neighbors")
plt.xlabel("n_neighbors")
plt.ylabel("Score")
plt.style.use('seaborn-poster')
plt.ylim(round(min(min(test_scores_mean),min(train_scores_mean)),1)-0.1, 1.1)
lw = 2

plt.plot(param_range, train_scores_mean, label="Training score", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, lw=lw)
plt.plot(param_range, test_scores_mean, label="5-fold cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig('Step5.1_KNeighborsClassifier_pca_ValiCurve.png')
plt.show()
optimum_alpha = param_range[list(test_scores_mean).index(max(test_scores_mean))]
print('Plot saved')
print('The maximum validation accuracy is:', max(test_scores_mean),'\n'
      'The corresponding traning accuracy is:', train_scores_mean[list(test_scores_mean).index(max(test_scores_mean))],'\n'
      'The optimum alpha is:',optimum_alpha)
clf = KNeighborsClassifier(n_neighbors = optimum_alpha )
clf = clf.fit(x_train, y_train)

print('The test accuracy is:',clf.score(x_test,y_test))


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf = clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = KNeighborsClassifier(n_neighbors = 2)
clf = clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = Perceptron()
clf = clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
