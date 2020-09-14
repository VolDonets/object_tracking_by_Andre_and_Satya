from sklearn.datasets import load_iris
from sklearn.tree import tree
from sklearn_porter import Porter

# Load data and train the classifier
saples = load_iris()
X, y = saples.data, saples.target
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

# Export
porter = Porter(clf, language='Java')
output = porter.export(embed_data=True)
print(output)