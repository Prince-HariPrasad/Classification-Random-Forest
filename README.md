# Classification-Random-Forest
This project demonstrates the use of the Random Forest algorithm to classify the Iris dataset.
# Iris Classification using Random Forest

This project demonstrates the use of the Random Forest algorithm to classify the Iris dataset.

## Steps

### Step 1: Import Libraries

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```
**Step 2: Load the Dataset**
```python
iris = load_iris()
X = iris.data
y = iris.target
```
**Step 3: Preprocess the Data**
(In this case, the data is already clean. Normally, preprocessing steps might include handling missing values, encoding categorical variables, etc.)

**Step 4: Train-Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
**Step 5: Train the Model**
```python
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
```
**Step 6: Evaluate the Model**
```python
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```
