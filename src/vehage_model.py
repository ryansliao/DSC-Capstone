import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def predict_vehicle_age(features, labels):
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

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