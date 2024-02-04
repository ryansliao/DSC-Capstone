import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
from sklearn.tree import DecisionTreeClassifier

def predict_destination(features, labels):
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    
    # Decision Tree Model 
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()[-3:]
    report_df.to_csv('outputs/vehtype_report.csv')