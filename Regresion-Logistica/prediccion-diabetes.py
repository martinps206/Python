import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Cargar datos
data = pd.read_csv('diabetes.csv')

# Preprocesamiento
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de regresión logística
logreg = LogisticRegression()
param_grid = {'C': [0.1, 1, 10], 'penalty': ['l2']}
grid = GridSearchCV(logreg, param_grid, cv=5)
grid.fit(X_train, y_train)

# Evaluación
y_pred = grid.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"ROC AUC: {roc_auc_score(y_test, grid.decision_function(X_test))}")

# Visualización
fpr, tpr, _ = roc_curve(y_test, grid.decision_function(X_test))
plt.plot(fpr, tpr, label='ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
