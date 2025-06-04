# main.py

import pandas as pd
import numpy as np
from src.utils import preprocess_data, split_data, cross_val_evaluate, exploratory_data_analysis, plot_confusion_matrix, plot_roc_curve 
from src.logistic_regression import LogisticRegression
from src.svm import SVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Cargar datos
red = pd.read_csv("data/winequality-red.csv", sep=";")
white = pd.read_csv("data/winequality-white.csv", sep=";")

# 2. Unir datasets y crear columna binaria 'target'
df = pd.concat([red, white], ignore_index=True)
df['target'] = np.where(df['quality'] >= 6, 1, 0)

# 3. Preprocesar y dividir
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# 4. Entrenar modelos
print("Entrenando Logistic Regression...")
lr = LogisticRegression(learning_rate=0.01, n_iters=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Entrenando SVM...")
svm = SVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# 5. Evaluar
def evaluate(y_true, y_pred, model_name):
    print(f"\nEvaluación para {model_name}:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))

evaluate(y_test, y_pred_lr, "Logistic Regression")
evaluate(y_test, y_pred_svm, "SVM")

print("Validación cruzada para Logistic Regression")
cross_val_evaluate(LogisticRegression, X, y, k=5, learning_rate=0.01, n_iters=1000)

#print("Validación cruzada para SVM")
#cross_val_evaluate(SVM, X, y, k=5, learning_rate=0.01, lambda_param=0.01, n_iters=1000)

#from sklearn.svm import SVC

# .

#print("\nEntrenando SVM con kernel RBF (Scikit-learn)...")
#svm_rbf = SVC(kernel='rbf', gamma='scale', C=1)
#svm_rbf.fit(X_train, y_train)
#y_pred_rbf = svm_rbf.predict(X_test)

#print("\nEvaluación para SVM con kernel RBF:")
#print("Accuracy:", accuracy_score(y_test, y_pred_rbf))
#print("Precision:", precision_score(y_test, y_pred_rbf))
#print("Recall:", recall_score(y_test, y_pred_rbf))
#print("F1 Score:", f1_score(y_test, y_pred_rbf))

# Llamar a análisis exploratorio (EDA)
exploratory_data_analysis(df)

# Mostrar matriz de confusión
#plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression")
#plot_confusion_matrix(y_test, y_pred_svm, "SVM")

# Mostrar curva ROC solo si tienes predict_proba en tu LogisticRegression
#if hasattr(lr, 'predict_proba'):
#    y_proba_lr = lr.predict_proba(X_test)
#    plot_roc_curve(y_test, y_proba_lr, "Logistic Regression")

from src.kernel_svm import KernelSVM
model = KernelSVM(kernel='rbf', C=1.0, gamma=0.05)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy Kernel SVM:", accuracy_score(y_test, y_pred))

from src.kernel_logistic import KernelLogisticRegression
model = KernelLogisticRegression(kernel='rbf', gamma=0.1, lr=0.01, epochs=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy Kernel Logistic Regression:", accuracy_score(y_test, y_pred))
