import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets

data_set = pd.read_csv('NIMS_file.csv')
data_set.head()

data_set.shape

data_set.isna().any()

"""# Data Preprocessing """

#different classes
protocoal_class=data_set.loc[:,"class"].values
protocoal_class

#count the number of classes
data_set['class'].value_counts()

#drop columns
data_set=data_set.drop(index=data_set[data_set['class']=='lime'].index)
data_set=data_set.drop(index=data_set[data_set['class']=='DNS'].index)
data_set=data_set.drop(index=data_set[data_set['class']=='HTTP'].index)
data_set=data_set.drop(index=data_set[data_set['class']=='shell'].index)
data_set=data_set.drop(index=data_set[data_set['class']=='sftp'].index)
data_set=data_set.drop(index=data_set[data_set['class']=='x11'].index)
data_set=data_set.drop(index=data_set[data_set['class']=='scp'].index)
data_set=data_set.drop(index=data_set[data_set['class']=='FTP'].index)
data_set=data_set.drop(index=data_set[data_set['class']=='TELNET'].index)
data_set.head()

data_set.columns

data_set.drop('total_bvolume',axis =1, inplace=True)
data_set.drop('total_fvolume',axis =1, inplace=True)
data_set.drop('id',axis =1, inplace=True)

data_set.shape

data_set['class'].value_counts()

data_set['new-class'] = pd.factorize(data_set['class'])[0] + 1
data_set = data_set.drop('class', 1)
data_set.head

"""#Training and testing - Naive Bayes Technique"""

#Extracting Independent and dependent Variable  
x = data_set.iloc[:,1:23].values
y = data_set.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,0] = le.fit_transform(x[:,0])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0) 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred  =  classifier.predict(X_test)

#Accuracy
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)
print(cm)
print(ac)
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

"""**Classification report**"""

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))

# visualize confusion matrix with seaborn heatmap
import seaborn as sns

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

"""#Claasication with Dynamic data - created by the CIC flowmeter"""

#Import library for comparision and processing
import numpy as np
import matplotlib.pyplot as plt

real_data=pd.read_csv('realtime_1.csv')
real_data.head()

real_data['Label'] = np.where(real_data['Src IP'].str[:3]== real_data['Dst IP'].str[:3], "localForwarding", "remoteForwarding")

#real_data['Src IP'] = real_data['Src IP'].str[:3]

real_data.shape

real_data.columns

#drop columns
real_data.drop(['Src IP', 'Dst IP','Timestamp','TotLen Fwd Pkts', 'TotLen Bwd Pkts','Flow ID', 'Src Port', 'Dst Port', 'Flow Byts/s',
       'Flow Pkts/s', 'Bwd IAT Tot','Fwd PSH Flags',
       'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Len',
       'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min',
       'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var',
       'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt',
       'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt',
       'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
       'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',
       'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
       'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts',
       'Subflow Bwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts',
       'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std',
       'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max',
       'Idle Min','Fwd IAT Tot','Flow IAT Mean','Flow IAT Std','Flow IAT Max', 'Flow IAT Min'], axis = 1,inplace=True)

real_data.columns

real_data['new-Label'] = pd.factorize(real_data['Label'])[0] + 1
real_data = real_data.drop('Label', 1)
real_data.head

real_data.shape

#total remote and local forwarding 
real_data['new-Label'].value_counts()

real_data.isnull()
real_data.isna().any()

# x attributes
X_real = real_data.iloc[:,:-1].values
X_real

# y attribute
y_real = real_data.iloc[:,20].values

"""**Prediction**"""

y_pred_real = classifier.predict(X_real)

print('y_pred_real=', y_pred_real)
print('y_real=', y_real)

#confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_real, y_pred_real))
print(confusion_matrix(y_real, y_pred_real))

"""**Comparition**"""

w=0.2
c_class = ["Naive Bayes","Naive Bayes with Real time"]

accuracy =[1,0.05]
precise = [1,1]
recall = [1,1]
F_score = [1,0.07]

bar1 = np.arange(len(accuracy))
bar2 = [i+w for i in bar1]
bar3 = [i+w for i in bar2]
bar4 = [i+w for i in bar3]

plt.bar(bar1, accuracy, w, label="accuracy")
plt.bar(bar2, precise, w, label="precise")
plt.bar(bar3, recall, w, label="recall")
plt.bar(bar4, F_score, w, label="F_score")

plt.xlabel("Naive Bayes classification")
plt.ylabel("Percentage")
plt.title("Comparison between NIMS and Real time dataset")

b= bar1+w+0.1
plt.xticks(b,c_class)

plt.yticks(np.arange(0,1.5,0.2))

plt.legend()
plt.show()

"""**Oversampling (for data balancing)**"""

from collections import Counter
from sklearn.datasets import make_classification
from imblearn import over_sampling

import seaborn as sns
from sklearn import preprocessing

from imblearn.over_sampling import RandomOverSampler
ros =RandomOverSampler(sampling_strategy="minority",random_state=0)

X_resampled,Y_resampled=ros.fit_resample(X_real, y_real)

print(sorted(Counter(Y_resampled).items()))

y_pred_resampled = classifier.predict(X_resampled)

print('y_pred_real=', y_pred_real)
print('y_pred_resampled=', y_pred_resampled)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_resampled,y_pred_resampled))
print(confusion_matrix(Y_resampled,y_pred_resampled))

"""#Comparision with NB, NB with real time, NB with balance dataset"""

w=0.2
c_class = ["Naive Bayes","Naive Bayes with Real time","Naive Bayes with Balance datset"]

accuracy =[1,0.16,0.51]
precise = [1,0.92,0.74]
recall = [1,0.99,0.99]
F_score = [1,0.25,0.67]

bar1 = np.arange(len(accuracy))
bar2 = [i+w for i in bar1]
bar3 = [i+w for i in bar2]
bar4 = [i+w for i in bar3]

plt.bar(bar1, accuracy, w, label="accuracy")
plt.bar(bar2, precise, w, label="precise")
plt.bar(bar3, recall, w, label="recall")
plt.bar(bar4, F_score, w, label="F_score")

plt.xlabel("Naive Bayes classification")
plt.ylabel("Percentage")
plt.title("Comparison of NIMS and Real time dataset")

b= bar1+w+0.1
plt.xticks(b,c_class)

plt.yticks(np.arange(0,1.5,0.2))

plt.legend()
plt.show()