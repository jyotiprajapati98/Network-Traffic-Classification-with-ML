

#**KNN simple**
"""#data preprocessing """

# importing libraries
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd

#importing datasets  
data_set= pd.read_csv('NIMS_file.csv') 
data_set.head()

#Extracting Independent and dependent Variable  
x= data_set.iloc[:, 1:23].values  
y= data_set.iloc[:, -1].values  
  
# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  
  
#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)

#Fitting K-NN classifier to the training set  
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))