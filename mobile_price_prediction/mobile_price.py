import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score

# Read data
train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

# Print the data info and description
train_set.info()
train_set.describe()

x_data = train_set.drop('price_range', axis=1)
x_data = StandardScaler().fit_transform(x_data)  # Standardize data
y_data = train_set['price_range']

columns = train_set.drop(['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi', 'price_range'],
                         axis=1).columns

figs, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 20))
axes = axes.flatten()

for i, col in enumerate(columns):
    sns.histplot(data=train_set, x=col, ax=axes[i], kde=True)
    axes[i].set_title(f"Histogram of {col}")

plt.tight_layout()
plt.show()

columns = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi", "price_range"]

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for i, col in enumerate(columns):
    sns.countplot(data=train_set, x=col, ax=axes[i], palette="Set2")
    axes[i].set_title(f"Countplot of {col}")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# One-hot encoding
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categorical_features = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"]
original_train = train_set.drop('price_range', axis=1)
encoded_train = one_hot_encoder.fit_transform(original_train[categorical_features])
train_df = pd.DataFrame(encoded_train, columns=one_hot_encoder.get_feature_names_out(categorical_features))
numerical_features_train = original_train.drop(categorical_features, axis=1)
x_data = pd.concat([numerical_features_train, train_df], axis=1)

# Standardize the data after one-hot encoding
x_data = StandardScaler().fit_transform(x_data)

# Split train set and evaluation set
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Dimensional reduction
pca = PCA(n_components=3, random_state=42)
x_data_pca = pca.fit_transform(x_data)

# Data Visualization
pca_x = x_data_pca[:, 0]
pca_y = x_data_pca[:, 1]
pca_z = x_data_pca[:, 2]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_x, pca_y, pca_z, c=y_data, cmap='viridis')

# Naive Bayes Classifier
start_time = time.perf_counter()
naive_bayes = GaussianNB()
naive_bayes.fit(x_train, y_train)
y_pred = naive_bayes.predict(x_test)
end_time = time.perf_counter()

# Evaluate model
print(f"Execution time: {end_time - start_time} seconds")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
print(classification_report(y_test, y_pred))

# Softmax Regression
start_time = time.perf_counter()
softmax = LogisticRegression(max_iter=200, solver='saga')
softmax.fit(x_train, y_train)
y_pred = softmax.predict(x_test)
end_time = time.perf_counter()

# Evaluate model
print(f"Execution time: {end_time - start_time} seconds")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
print(classification_report(y_test, y_pred))

# Support Vector Machine
start_time = time.perf_counter()
svm = SVC(kernel='linear', C=10)  # Soft margins
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
end_time = time.perf_counter()

# Evaluate model
print(f"Execution time: {end_time - start_time} seconds")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
print(classification_report(y_test, y_pred))

# Multi Layers Perceptron
start_time = time.perf_counter()
ann = MLPClassifier(hidden_layer_sizes=(100), max_iter=1000, random_state=42)
ann.fit(x_train, y_train)
y_pred = ann.predict(x_test)
end_time = time.perf_counter()

# Evaluate model
print(f"Execution time: {end_time - start_time} seconds")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
print(classification_report(y_test, y_pred))

test_set.info()
test_set.describe()

test_id = test_set['id']
x_test_set = test_set.drop('id', axis=1)
x_test_set = StandardScaler().fit_transform(x_test_set)

columns = test_set.drop(['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi'], axis=1).columns

figs, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 20))
axes = axes.flatten()

for i, col in enumerate(columns):
    sns.histplot(data=test_set, x=col, ax=axes[i], kde=True)
    axes[i].set_title(f"Histogram of {col}")

plt.tight_layout()
plt.show()

columns = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(columns):
    sns.countplot(data=test_set, x=col, ax=axes[i], palette="Set2")
    axes[i].set_title(f"Countplot of {col}")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# One-hot encoding
original_test = test_set.drop('id', axis=1)
encoded_test = one_hot_encoder.transform(original_test[categorical_features])
test_df = pd.DataFrame(encoded_test, columns=one_hot_encoder.get_feature_names_out(categorical_features))
numerical_features_test = original_test.drop(categorical_features, axis=1)
x_test_set = pd.concat([numerical_features_test, test_df], axis=1)

# Standardize the data after one-hot encoding
x_test_set = StandardScaler().fit_transform(x_test_set)

# Dimensional reduction
x_test_set_pca = pca.fit_transform(x_test_set)

# Clustering data
gmm = GaussianMixture(n_components=4, random_state=42)
clusters = gmm.fit_predict(x_test_set_pca)

# Data Visualization
pca_x = x_test_set_pca[:, 0]
pca_y = x_test_set_pca[:, 1]
pca_z = x_test_set_pca[:, 2]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_x, pca_y, pca_z, c=clusters, cmap='viridis')

y_predict = softmax.predict(x_test_set)

for id, res in zip(test_id, y_predict):
    if res == 0:
        res = "Low cost"
    elif res == 1:
        res = "Medium cost"
    elif res == 2:
        res = "High cost"
    else:
        res = "Very High cost"
    print(f"Phone {id} has {res} price")
