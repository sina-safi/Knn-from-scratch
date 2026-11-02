import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

wine = load_wine()
X = wine.data
y = wine.target

print("Samples:", X.shape[0], "| Features:", X.shape[1])
print("Missing values:", np.isnan(X).sum())

# split: 64% train, 16% val, 20% test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# outlier detection with IQR (fit on train only)
Q1 = np.percentile(X_train, 25, axis=0)
Q3 = np.percentile(X_train, 75, axis=0)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_per_feature = ((X_train < lower_bound) | (X_train > upper_bound)).sum(axis=0)
for feature, count in zip(wine.feature_names, outliers_per_feature):
    print(f"{feature}: {count}")

# winsorize + standardize (train stats only)
X_train_w = np.clip(X_train, lower_bound, upper_bound)
X_val_w   = np.clip(X_val,   lower_bound, upper_bound)
X_test_w  = np.clip(X_test,  lower_bound, upper_bound)

mean = X_train_w.mean(axis=0)
std  = X_train_w.std(axis=0)
std[std == 0] = 1.0

X_train_s = (X_train_w - mean) / std
X_val_s   = (X_val_w   - mean) / std
X_test_s  = (X_test_w  - mean) / std

def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def Neighbors(X_train, y_train, x_test, k):
    distances = [(distance(x, x_test), y) for x, y in zip(X_train, y_train)]
    distances.sort()
    return distances[:k]

def Predict(X_train, y_train, x_test, k):
    neighbors = Neighbors(X_train, y_train, x_test, k)
    votes = Counter(label for d, label in neighbors)
    return votes.most_common(1)[0][0]

def predict(X_train, y_train, X_test, k):
    return np.array([Predict(X_train, y_train, x, k) for x in X_test])

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# tune k on validation
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
results = []
for k in k_values:
    acc = accuracy(y_val, predict(X_train_s, y_train, X_val_s, k))
    results.append(acc)
    print(f"k={k}, val acc={acc:.4f}")

best_k = k_values[int(np.argmax(results))]
print(f"best k = {best_k}")

plt.plot(k_values, results, marker="o")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k")
plt.savefig("accuracy_vs_k.png", dpi=150)
plt.show()

# final eval on test
y_test_pred = predict(X_train_s, y_train, X_test_s, best_k)
print(f"\nTest accuracy (k={best_k}): {accuracy(y_test, y_test_pred):.4f}")
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, target_names=wine.target_names))

# compare with sklearn
sk_knn = KNeighborsClassifier(n_neighbors=best_k)
sk_knn.fit(X_train_s, y_train)
print(f"sklearn accuracy: {accuracy(y_test, sk_knn.predict(X_test_s)):.4f}")