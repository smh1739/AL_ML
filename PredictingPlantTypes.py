#Import the necessary libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

iris = None
X = None
y = None
target_names = None

iris = load_iris()
X = iris.data[:, :2]
y = iris.target
target_names = iris.target_names

mean = None
std = None

scaler = StandardScaler()
X_normalised = scaler.fit_transform(X)

mean = scaler.mean_
std = scaler.scale_
print("Mean (µ) used for normalisation:", mean)
print("Standard deviation (σ) used for normalisation:", std)

knn = None
y_pred = None
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_normalised, y)
y_pred = knn.predict(X_normalised)

random_indices = np.random.choice(len(X), size=5, replace=False)

print("\nSample predictions (5 random instances):")
for i in random_indices:
    print(f"Sepal length: {X[i,0]:.2f}, Sepal width: {X[i,1]:.2f} | "
          f"True class: {target_names[y[i]]} | Predicted class: {target_names[y_pred[i]]}")

def predict_new_samples(new_samples):
    """
    new_samples: numpy array of shape (n_samples, 2) with sepal length and width
    new_samples_norm to be computed from scaler.mean_ and scaler.scale_
    predictions: numpy array of shape (n_samples,) with predicted class indices

    """
    return None

def predict_new_samples(new_samples):
    new_samples_norm = scaler.transform(new_samples)
    predictions = knn.predict(new_samples_norm)
    return [target_names[p] for p in predictions]

new_data = np.array([[5.0, 3.5], [6.5, 3.0]])
predicted_classes = predict_new_samples(new_data)
print("\nPredictions for new samples:")
for sample, pred in zip(new_data, predicted_classes):
    print(f"Sepal length: {sample[0]}, Sepal width: {sample[1]} => Predicted class: {pred}")

### GRADED CELL
iris_full = ...
X_full = ...
y_full = ...
target_names = ...
scaler = ...
X_full_normalised = ...
knn = ...
unknown_sample = np.array([[5.8, 2.7, 5.1, 1.9]])
unknown_sample_normalised = ...
predicted_class_index = ...
predicted_class_name = ...


iris_full = load_iris()
X_full = iris_full.data
y_full = iris_full.target
target_names = iris_full.target_names
scaler = StandardScaler()
X_full_normalised = scaler.fit_transform(X_full)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_full_normalised, y_full)
unknown_sample_normalised = scaler.transform(unknown_sample)
predicted_class_index = knn.predict(unknown_sample_normalised)[0]
predicted_class_name = target_names[predicted_class_index]
print(f"Predicted class for the unknown sample {unknown_sample[0]} is: {predicted_class_name}")
