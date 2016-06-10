# import a dataset
from sklearn import datasets  # , tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
# from sklearn.neighbors import KNeighborsClassifier
# import random


def euc(a, b):
    return distance.euclidean(a, b)


class ScrappyKNN():
    '''My fist classifier.

    Just reminder:
        X_train - training features
        y_train - training labels
        X_test -  test features
        y_test -  test labels

    The accuracy can change beacuse of randomicity of dataset.
    '''

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

# my_classifier = tree.DecisionTreeClassifier()
# my_classifier = KNeighborsClassifier()
my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

print(accuracy_score(y_test, predictions))
