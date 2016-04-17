from sklearn import tree

# 1 smooth, 0 bumpy
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# 0 apple, 1 orange
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[160, 0]]))
