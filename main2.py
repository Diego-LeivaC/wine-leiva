import pandas as pd
import numpy as np
from src.utils import preprocess_data, split_data
from src.logistic_regression import LogisticRegression
from src.svm import SVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_and_prepare_data():
    # Cargar datasets
    red = pd.read_csv('data/winequality-red.csv', sep=';')
    white = pd.read_csv('data/winequality-white.csv', sep=';')
    
    # Añadir una columna para diferenciar los tipos (opcional)
    red['type'] = 0
    white['type'] = 1
    
    # Combinar los datasets
    data = pd.concat([red, white], axis=0)
    
    # Crear variable binaria
    data['target'] = (data['quality'] >= 6).astype(int)
    
    return data

def main():
    print("Loading and preprocessing data...")
    data = load_and_prepare_data()
    
    # Preprocesar
    X, y = preprocess_data(data)  # deberás definir esto en src/utils.py
    X_train, X_test, y_train, y_test = split_data(X, y)  # también en utils

    print("Training Logistic Regression...")
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    print("Training SVM...")
    svm_model = SVM()
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)

    # Evaluación
    print("\nLogistic Regression Results:")
    print_metrics(y_test, y_pred_lr)

    print("\nSVM Results:")
    print_metrics(y_test, y_pred_svm)

def print_metrics(y_true, y_pred):
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
    main()
