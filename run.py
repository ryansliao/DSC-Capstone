import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier

from etl import get_data


# Predicting Vehicle Type
def preprocess_vehicle_type(vehicles):
    # Filter out missing/uesless data.
    vehicles_type_df = vehicles[(vehicles['VEHTYPE'] > 0)
                                & (vehicles['VEHTYPE'] != 6)
                                & (vehicles['VEHTYPE'] != 97)
                                & (vehicles['HTPPOPDN'] != -9)
                                & (vehicles['CAR'] > 0)
                                & (vehicles['PRICE'] > 0)
                                & (vehicles['PLACE'] > 0)
                                & (vehicles['GSCOST'] != -9)
                                & (vehicles['URBAN'] != 4)]

    # Rename classes.
    vehicles_type_df = vehicles_type_df.astype(object)
    vehicles_type_df.loc[vehicles_type_df['VEHTYPE'] == 1, 'VEHTYPE'] = 'Automobile'
    vehicles_type_df.loc[vehicles_type_df['VEHTYPE'] == 2, 'VEHTYPE'] = 'Van'
    vehicles_type_df.loc[vehicles_type_df['VEHTYPE'] == 3, 'VEHTYPE'] = 'SUV'
    vehicles_type_df.loc[vehicles_type_df['VEHTYPE'] == 4, 'VEHTYPE'] = 'Truck'
    vehicles_type_df.loc[vehicles_type_df['VEHTYPE'] == 7, 'VEHTYPE'] = 'Motorcycle'
    return vehicles_type_df

def predict_vehicle_type(vehicles_type_df):
    # Split data
    X = vehicles_type_df[['CNTTDHH', 'CAR', 'CDIVMSAR', 'GSCOST', 'HTPPOPDN', 'HHFAMINC', 'HHSIZE', 'HHVEHCNT', 'PRICE', 'PLACE']]
    y = vehicles_type_df['VEHTYPE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
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


# Predicting Vehicle Fuel Type
def preprocess_vehicle_fuel(vehicles):
    vehicles['FUELTYPE'] = vehicles['FUELTYPE'].replace(6, 4)
    vehicles_fuel_df = vehicles[(vehicles['FUELTYPE'] > 0)
                                & (vehicles['FUELTYPE'] != 97)
                                & (vehicles['HTPPOPDN'] != -9)
                                & (vehicles['HHFAMINC'] > 0)
                                & (vehicles['CAR'] > 0)
                                & (vehicles['PRICE'] > 0)
                                & (vehicles['PLACE'] > 0)
                                & (vehicles['URBAN'] != 4)]
    vehicles_fuel_df = vehicles_fuel_df.astype(object)
    vehicles_fuel_df.loc[vehicles_fuel_df['FUELTYPE'] == 1, 'FUELTYPE'] = 'Gas'
    vehicles_fuel_df.loc[vehicles_fuel_df['FUELTYPE'] == 2, 'FUELTYPE'] = 'Diesel'
    vehicles_fuel_df.loc[vehicles_fuel_df['FUELTYPE'] == 3, 'FUELTYPE'] = 'Hybrid/Electric'
    return vehicles_fuel_df

def predict_vehicle_fuel(vehicles_fuel_df):
    X = vehicles_fuel_df[['CAR', 'CNTTDHH', 'CDIVMSAR', 'HTPPOPDN', 'HHSIZE', 'HHFAMINC', 'PRICE', 'PLACE']]
    y = vehicles_fuel_df['FUELTYPE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Random Forest Model
    rf_model = RandomForestClassifier(class_weight='balanced')
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('outputs/vehfuel_report.csv')

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Diesel', 'Gas', 'Hybrid/Electric']).plot()
    plt.title('Vehicle Fuel Random Forest Confusion Matrix')
    plt.savefig('outputs/vehfuel_matrix.png')

    y_prob = rf_model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=['Diesel', 'Gas', 'Hybrid/Electric'])

    # Compute precision-recall curve for each class
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(1, 4):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i-1], y_prob[:, i-1])
        average_precision[i] = average_precision_score(y_test_bin[:, i-1], y_prob[:, i-1])

    # Plot the precision-recall curves
    plt.figure(figsize=(7, 7))

    for i in range(1, 4):
        plt.plot(recall[i], precision[i], label=f'Class {i} (AP={average_precision[i]:0.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Vehicle Fuel Random Forest')
    plt.legend()
    plt.savefig('outputs/vehfuel_prc.png')


# Predicting Vehicle Age
def preprocess_vehicle_age(vehicles):
    vehicles_age_df = vehicles[(vehicles['VEHAGE'] < 40)
                            & (vehicles['VEHAGE'] > 0)
                            & (vehicles['HTPPOPDN'] != -9)
                            & (vehicles['HHFAMINC'] > 0)
                            & (vehicles['CAR'] > 0)
                            & (vehicles['PRICE'] > 0)
                            & (vehicles['PLACE'] > 0)
                            & (vehicles['URBAN'] != 4)]
    return vehicles_age_df

def predict_vehicle_age(vehicles_age_df):
    X = vehicles_age_df[['CAR', 'CNTTDHH', 'CDIVMSAR', 'GSCOST', 'HHSIZE', 'HHFAMINC', 'HHVEHCNT', 'PRICE']]
    y = vehicles_age_df['VEHAGE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_pred = lr_model.predict(X_test)

    lr_pred = pd.DataFrame({'Linear Regression Predictions': y_pred, 'Testing Data': y_test})
 
    # drawing the plot
    sns.set(font_scale=1.1)
    sns.lmplot(x="Linear Regression Predictions", y="Testing Data", data=lr_pred)
    plt.savefig('outputs/vehage_lr.png')

    # Relative Root Mean Squared Error
    def relative_root_mean_squared_error(true, pred):
        num = np.sum(np.square(true - pred))
        den = np.sum(np.square(pred))
        squared_error = num/den
        rrmse_loss = np.sqrt(squared_error)
        return rrmse_loss
    
    rrmse = relative_root_mean_squared_error(y_test, y_pred)

    with open('outputs/vehage_rrmse.txt', 'w') as f:
        f.write('Vehicle Age Linear Regression RRMSE: {}'.format(rrmse))
        f.close()


# Main Function

if __name__=="__main__": 

    vehicles = get_data()

    print("Running Vehicle Type Model...")
    vehicles_type_df = preprocess_vehicle_type(vehicles)
    predict_vehicle_type(vehicles_type_df)

    print("Running Vehicle Fuel Model...")
    vehicles_fuel_df = preprocess_vehicle_fuel(vehicles)
    predict_vehicle_fuel(vehicles_fuel_df)

    print("Running Vehicle Age Model...")
    vehicles_age_df = preprocess_vehicle_age(vehicles)
    predict_vehicle_age(vehicles_age_df)

    print("All models have finished running!")