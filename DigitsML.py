from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle

X, y = fetch_openml('mnist_784', return_X_y=True, as_frame=False)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
scaled_train_X = scaler.fit_transform(train_X)
scaled_test_X = scaler.transform(test_X)

model = SVC(kernel='linear', C=1, random_state=42, gamma=0.3, max_iter=5000, verbose=1)
model.fit(scaled_train_X, train_y)

y_pred = model.predict(scaled_test_X)
print(classification_report(test_y, y_pred))