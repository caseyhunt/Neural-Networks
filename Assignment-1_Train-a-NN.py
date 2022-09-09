# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:48:55 2022

@author: casey hunt
"""

from sklearn.datasets import load_wine
wine = load_wine()
x = wine.data
y = wine.target


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 15, test_size=.3)
x_train_subsplit, x_junk, y_train_subsplit, y_junk = train_test_split(x_train, y_train, random_state = 15, train_size=.2)


from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
# stdsc.fit(x_train)
stdsc.fit(x_train_subsplit)

# x_train_std = stdsc.transform(x_train)
x_train_subsplit_std = stdsc.transform(x_train_subsplit)
x_test_std = stdsc.transform(x_test)
print(len(x_train_subsplit_std))


from sklearn.neural_network import MLPClassifier

n_hidden_nodes = 10
mlp = MLPClassifier(activation ='tanh', max_iter = 40, random_state = 15, verbose = True, learning_rate_init = 0.01, hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes])
mlp.fit(x_train_subsplit_std, y_train_subsplit)

print("Activaton Function: {}".format(mlp.activation))
print("List of predicted classes: {}".format(mlp.classes_))
print("Training set loss: {}".format(mlp.loss_))


from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(1)
plt.plot(mlp.loss_curve_)
plt.xlabel("Epochs")
plt.ylabel("Loss")


plt.figure(2)
y_predicted = mlp.predict(x_test_std)
print(metrics.classification_report(y_test, y_predicted))
mat = metrics.confusion_matrix(y_test, y_predicted)
sns.heatmap(mat.T, square = True, annot = True, fmt = "d", cbar = False)
plt.xlabel("True Label")
plt.ylabel("Predicted Label")


