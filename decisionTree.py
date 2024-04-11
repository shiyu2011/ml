from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt


#load dataset
iris = load_iris()
X, y = iris.data, iris.target

#split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#initialize and train the decision tree classfier
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

#predict and ealuation

#visualize the trained Decision tree
plt.figure(figsize=(12,8))
tree.plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True)
plt.show()