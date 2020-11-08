from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
# X, y = load_iris(return_X_y=True)
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X, y)
# tree.plot_tree(clf)
# dot_data = tree.export_graphviz(clf, out_file=None) 
# graph = graphviz.Source(dot_data) 
# graph.render("iris")

MARS = 1
VENUS = 0

features = [
    [1, 1, 1, 1],
    [0, 1, 1, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 1, 1]
]

labels = [
    MARS,
    VENUS,
    VENUS,
    VENUS,
    VENUS,
    MARS,
    MARS,
    MARS,
    MARS,
    VENUS
]

classification = tree.DecisionTreeClassifier()
classification = classification.fit(features, labels)

tree.plot_tree(classification)

dot_data = tree.export_graphviz(classification, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("aliens")