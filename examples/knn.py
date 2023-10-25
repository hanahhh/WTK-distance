import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from ot_dis.wtk import wtk_distance
from ot_dis.otw import otw_distance
from ot_dis.kpg import kpg_sequence_distance, kpg_2d_rl_kp
# from wtk import wtk_distance
# from otw import otw_distance, otw_distance_1
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
    return accuracy

def knn_OTW(X_train, X_test, y_train, y_test, m, s, k=1):
    train_size = len(X_train)
    test_size = len(X_test)
    result = np.zeros((test_size, train_size))
    for train_idx in tqdm(range(train_size)):
        for test_idx in tqdm(range(test_size), leave=False):
            distance = otw_distance(X_train[train_idx], X_test[test_idx], m, s)
            result[test_idx, train_idx] = distance
    print(f"_______________{result.shape}_______________")
    y_pred = knn_classifier_from_distance_matrix(
        distance_matrix=result,
        k=k,
        labels=y_train,
    )
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def knn_sequence_KPG(X_train, X_test, y_train, y_test, method="2d_normal",lamb=3, sub_length=25, k=1):
    kpg_dict = {
        "1d_sequence": kpg_sequence_distance,
        "2d_normal": kpg_2d_rl_kp,
    }
    train_size = len(X_train)
    test_size = len(X_test)
    result = np.zeros((test_size, train_size))
    for train_idx in tqdm(range(train_size)):
        for test_idx in tqdm(range(test_size), leave=False):
            distance = kpg_dict[method](X_train[train_idx], X_test[test_idx], lamb=lamb, sub_length=sub_length)
            result[test_idx, train_idx] = distance
    
    y_pred = knn_classifier_from_distance_matrix(
        distance_matrix=result,
        k=k,
        labels=y_train,
    )
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy