
import numpy as np
import pandas as pd
from lreg import LogisticRegression

test_length = 114 # for breast cancer
nofeats = 30 # for breast cancer

#np.random.seed(1)

# -----------------------------------------

# data:

data = pd.read_csv('bc_std.csv')

data = np.array(data)
m,n = data.shape

np.random.shuffle(data)

data_test = data[0:test_length]
X_test = data_test[:,0:nofeats]
Y_test = data_test[:,nofeats]
Y_test = Y_test.T

data_train = data[test_length:m]
X_train = data_train[:, 0:nofeats] 
Y_train = data_train[:,nofeats] 
Y_train = Y_train.T

#print(X_train.shape, Y_train.shape)

# -----------------------------------------

def accuracy(y_pred, y_test):
	return np.sum(y_pred==y_test)/len(y_test)

classifier = LogisticRegression()
classifier.fit(X_train , Y_train)
y_pred = classifier.predict(X_test)

#print("real :",Y_test)
#print("predicts: ",y_pred)

acc = accuracy(y_pred, Y_test)
print("accuracy = ")
print(acc)


