

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA 
path='/SOT/'
train_df=pd.read_csv(path+"train_1000.csv")

#test the data accuracy
stat=train_df['Sentiment'].value_counts()
print(stat)
train=np.array(train_df)
X=train[:,2:]
print(X.shape)
#See the variance of all dimension to choice the number of dimension.

  
pca = PCA(copy=True, n_components=None, whiten=False)
pca.fit(X)  
var=pca.explained_variance_ratio_
PCA_ANALYSIS=[]
cul=0
for i in range(len(var)):
    cul=cul+var[i]
    PCA_ANALYSIS=[[i,var[i],cul]]
    print(i,cul)

PCA_ANALYSIS=DataFrame(PCA_ANALYSIS,columns=['component','var','cumulative_var'])
PCA_ANALYSIS.to_csv(path+'PCA_RESULTS.csv',index_label ='ID',header=True)
print('PCA results saved to', path)

train[:,1].sum()
print(X.shape)
pca = PCA(copy=True, n_components=369, whiten=False) 
pca.fit(X)  

#PCA(copy=True, n_components=None, whiten=False)  
var=pca.explained_variance_ratio_
print(len(var))

print(var.shape)
X_DR=pca.transform(X)
print(X_DR.shape)

col_name=[]
for i in range(1,len(X_DR[0])+1):
    col_name.append('X'+str(i))
len(col_name)
from pandas import DataFrame
df_DR=DataFrame(X_DR,columns=col_name)
df_DR['sentiment']=train[:,1]
df_DR = df_DR[['sentiment']+col_name]

df_DR.to_csv(path+'train_1000_DR.csv',index_label ='ID',header=True)
print('DR train data saved to', path)

#split dataset to training set and test dataset for some models of SAS. If you do not need to use these SAS model, ignore this par.
df_DR_train = df_DR[:800]
df_DR_train.to_csv(path+'BIA652_train_1000_DR_train.csv',index_label ='ID',header=True)
print(df_DR_train.shape)

df_DR_test = df_DR[800:]
df_DR_test.to_csv(path+'BIA652_test_1000_DR_test.csv',index_label ='ID',header=True)
print(df_DR_test.shape)
