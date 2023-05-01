import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, X_test):
        return np.sqrt(np.sum((self.X_train - X_test)**2, axis=1))

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            # print(x)
            distances = self.euclidean_distance(x)
            indices = np.argsort(distances)[:self.k]
            k_nearest_classes = [self.y_train[i] for i in indices]
            most_common_class = Counter(k_nearest_classes).most_common(1)[0][0]
            y_pred.append(most_common_class)
        return np.array(y_pred)