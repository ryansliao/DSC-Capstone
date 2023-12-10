import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier

def predict_vehicle_type(features, labels):
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    
    # Random Forest Model 
    rf_model = RandomForestClassifier(class_weight='balanced')
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('outputs/vehtype_report.csv')

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Automobile', 'Motorcycle', 'SUV', 'Truck', 'Van']).plot()
    plt.title('Vehicle Type Random Forest Confusion Matrix')
    plt.savefig('outputs/vehtype_matrix.png')

    y_prob = rf_model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=['Automobile', 'Motorcycle', 'SUV', 'Truck', 'Van'])

    # Compute precision-recall curve for each class
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(1, 6):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i-1], y_prob[:, i-1])
        average_precision[i] = average_precision_score(y_test_bin[:, i-1], y_prob[:, i-1])

    # Plot the precision-recall curves
    plt.figure(figsize=(7, 7))

    for i in range(1, 6):
        plt.plot(recall[i], precision[i], label=f'Class {i} (AP={average_precision[i]:0.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Vehicle Type Random Forest')
    plt.legend()
    plt.savefig('outputs/vehtype_prc.png')
