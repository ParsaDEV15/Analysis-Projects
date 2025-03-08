from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report


#Not Completed
class CustomBoosting:
    def __init__(self, n_estimators, learning_rate, max_iter):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples
        self.classes = np.unique(y)

        for _ in range(self.n_estimators):
            params = {'kernel': ['linear', 'poly', 'rbf'], 'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
            model = GridSearchCV(SVC(), param_grid=params, scoring='accuracy', n_jobs=-1)
            model.fit(X, y, sample_weight=sample_weights)
            y_pred = model.predict(X)

            error = np.sum(sample_weights * (y_pred != y)) / np.sum(sample_weights) + 1e-10
            alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)

            self.models.append(model.best_estimator_)
            self.alphas.append(alpha)

            sample_weights *= np.exp(alpha * (y_pred != y))
            sample_weights /= np.sum(sample_weights)

            print(f"Model {_+1}: Error = {error:.4f}, Alpha = {alpha:.4f}")

    def predict(self, X):
        # classes_score = np.zeros((X.shape[0], len(self.classes)))
        # for model, alpha in zip(self.models, self.alphas):
        #     y_pred = model.predict(X)
        #     for i, cls in enumerate(self.classes):
        #         classes_score[:, i] += alpha * (y_pred == cls)
        # return self.classes[np.argmax(classes_score, axis=1)]

        best_model = max(zip(self.models, self.alphas), key=lambda x: x[1])
        print(f"Best model: {best_model}")
        # y_pred = best_model.predict(X)
        # return y_pred


if __name__ == "__main__":
    X, y = make_blobs(n_samples=150, centers=3, cluster_std=3, random_state=128)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    booster = CustomBoosting(n_estimators=30, learning_rate=1, max_iter=500)
    booster.fit(X_train, y_train)

    y_pred = booster.predict(X_test)
    # print("Predicted labels:", y_pred)
    # print("True labels:", y_test)
    # print(classification_report(y_test, y_pred, zero_division=0))