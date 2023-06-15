# training pipeline in pytorch:
#
#1) design model (input, output size, forward pass)
#2) construct loss and optimizer
#3) training loop
#	- forward pass : compute prediction and loss
#	- backward pass : gradients
#	- update weights

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# -------------------------------------------------
# data

#np.random.seed(1)
test_length = 114

data = pd.read_csv('bc_std.csv')

data = np.array(data)
n_samples, n_features = data.shape
n_features = n_features -1
print(n_samples, n_features)

np.random.shuffle(data)

data_test = data[0:test_length]
X_test = data_test[:,0:n_features]
y_test = data_test[:,n_features]
y_test = y_test.T

data_train = data[test_length:n_samples]
X_train = data_train[:, 0:n_features] 
y_train = data_train[:,n_features] 
y_train = y_train.T


# convert to torch tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# reshape target tensor
y_train = y_train.view(y_train.shape[0],1) # make it column
y_test = y_test.view(y_test.shape[0],1) # make it column

# -------------------------------------------------
# model
# f = wx+b, sigmoid at the end (for probability to fall into each class)

class LogisticRegression(nn.Module):
	def __init__(self, n_input_features):
		super(LogisticRegression, self).__init__()
		self.linear = nn.Linear(n_input_features, 1)

	def forward(self,x):
		y_pred = torch.sigmoid(self.linear(x))
		return y_pred
		
model = LogisticRegression(n_features)
print(model)

# loss and optimizer
learning_rate = 0.1
criterion = nn.BCELoss() # binary cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# training loop
num_epochs = 1000
for epoch in range(num_epochs):
	# forward pass and loss
	y_pred = model(X_train)
	loss = criterion(y_pred, y_train)

	# backward pass
	loss.backward()
	
	# update weights
	optimizer.step()
	
	# empty gradients (zero gradients) so not summed up
	optimizer.zero_grad()
	
	if epoch % 100 ==0:
		print(f'epoch: {epoch}, loss = {loss.item():.4f}')
		
# evaluation:
# evaluation should be outside the computational graph
with torch.no_grad():
	y_pred = model(X_test)
	y_pred_cls = y_pred.round()
	acc = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])
	print(f'accuracy = {acc:.4f}')



