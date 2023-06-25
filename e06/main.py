import os
import struct
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def read_mfcc_file(file_path):
    """The first 4 bytes of the binary file is an integer indicating the total number
    of data points in the file. This is used by the program that reads this file to
    check the endianness of the file by comparing with the file size. The rest
    of the file is simply the data points. Each data point is a 4-byte floating
    point number, in big-endian order.
    """
    with open(file_path, 'rb') as f:
        num_data_points_bytes = f.read(4)
        num_data_points = struct.unpack('>i', num_data_points_bytes)[0]
        data_bytes = f.read()
        data = [struct.unpack('>f', data_bytes[i:i+4])[0] for i in range(0, len(data_bytes), 4)]
        feature_vectors = [data[i:i+13] for i in range(0, len(data), 13)]
        return feature_vectors

def extract_label(file_name):
    """extracts the filename without the filename extension"""
    return file_name[0]

def get_file_path(file_name):
    """total path of data file"""
    base_path = r'e06/data/CEP'
    return os.path.join(base_path, file_name)

def read_file_list(file_path):
    """reads the filenames to be processed"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def normalize(vector_list, means, stds):
    """normalizes a list of vectors component-wise using means and standard deviations"""
    temp = []
    for vec in vector_list:
        vec = np.asarray(vec)
        normalized_vec = (vec - means) / stds
        temp.append(normalized_vec)
    return temp

training_files = read_file_list(r'e06/data/Listen/trainingSarah.txt')
test_files = read_file_list(r'e06/data/Listen/testCEP.txt')

# preprocessing data
X_train = []
y_train = []
for file_name in training_files:
    file_path = get_file_path(file_name)
    feature_vectors = read_mfcc_file(file_path)
    middle_vector = feature_vectors[len(feature_vectors) // 2]
    X_train.append(middle_vector)
    y_train.append(extract_label(file_name))

X_train = np.array(X_train)

# calculate mean and standard deviation for each component
means = np.mean(X_train, axis=0)
stds = np.std(X_train, axis=0)

# normalize training data
X_train_normalized = normalize(X_train, means, stds)

# train knn classifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train_normalized, y_train)

# process test data with knn model
X_test = []
y_true = []
for file_name in test_files:
    file_path = get_file_path(file_name)
    feature_vectors = read_mfcc_file(file_path)
    middle_vector = feature_vectors[len(feature_vectors) // 2]
    X_test.append(middle_vector)
    y_true.append(extract_label(file_name))

X_test_normalized = normalize(X_test, means, stds)

y_pred = neigh.predict(X_test_normalized)

# calculate accuracy
accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
print(f'The sound recognition rate of test sample is {accuracy * 100:.2f}%')