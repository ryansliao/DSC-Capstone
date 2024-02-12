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
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify = y)

    # Adjust for class imbalance
    num_samples = int(np.ceil(y_train.value_counts().median()))
    sampling_strategy = {label: num_samples if count < num_samples else count for label, count in Counter(y_train).items()}
    smote = SMOTE(k_neighbors = 4, sampling_strategy = sampling_strategy)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Decision Tree Model 
    model = DecisionTreeClassifier()
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_resampled)

    # Calculate traning and test accuracy
    train_accuracy = accuracy_score(y_resampled, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    performance_str = ""
    performance_str += "Accuracy on Train Set: {:.2f}%".format(train_accuracy * 100)
    performance_str = "\n"
    performance_str += "Accuracy on Test Set: {:.2f}%".format(test_accuracy* 100)
    performance_str = "\n"

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_test, y_pred, average = 'macro')
    recall = recall_score(y_test, y_pred, average = 'macro')
    f1 = f1_score(y_test, y_pred, average = 'macro')
    performance_str += "Precision on Test Set: {:.2f}%".format(precision * 100)
    performance_str = "\n"
    performance_str += "Recall on Test Set: {:.2f}%".format(recall * 100)
    performance_str = "\n"
    performance_str += "F1 Score on Test Set: {:.2f}%".format(f1 * 100)

    # Generating performance report
    performance = open("performance.txt", "w")
    performance.write(performance_str)
    performance.close()

    # Generating classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()[-3:]
    report_df.to_csv('outputs/vehtype_report.csv')