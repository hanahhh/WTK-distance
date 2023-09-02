import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from wtk import wtk_distance
import time

def knn_classifier_from_distance_matrix(distance_matrix, k, labels):
    knn_clf = KNeighborsClassifier(
        n_neighbors=k, algorithm="brute", metric="precomputed"
    )
    n_train_samples = distance_matrix.shape[1]
    knn_clf.fit(np.random.rand(n_train_samples, n_train_samples), labels)
    predicted_labels = knn_clf.predict(distance_matrix)
    return predicted_labels

def knn_WTK(X_train, X_test, y_train, y_test, data_set, sub_length,k=3):
    start_time = time.time()
    train_size = len(X_train)
    test_size = len(X_test)
    result = np.zeros((test_size, train_size))
    for train_idx in tqdm(range(train_size)):
        for test_idx in tqdm(range(test_size), leave=False):
            distance = wtk_distance(X_train[train_idx], X_test[test_idx], sub_length)
            result[test_idx, train_idx] = distance
    
    y_pred = knn_classifier_from_distance_matrix(
        distance_matrix=result,
        k=k,
        labels=y_train,
    )
    accuracy = accuracy_score(y_test, y_pred)
    end_time = time.time()
    # with open('experiment.txt', 'a') as file:
    #     file.write(f"Data set: {data_set} - Accuracy: {accuracy} - Subsequence length: {sub_length} - k neighbors: {k} - execution time: {end_time - start_time}\n")
    return accuracy