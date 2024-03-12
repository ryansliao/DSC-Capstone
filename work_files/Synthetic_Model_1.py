#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, r2_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
import logging


# In[2]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[3]:


logging.basicConfig(filename='log.txt', 
                    filemode='a', 
                    level=logging.INFO,
                    datefmt='%H:%M:%S',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s')


# In[4]:


feature_df = pd.read_csv('feature_df.csv')
feature_df.head()


# In[5]:


# merge district with household TAZ
district_df = pd.read_csv('taz_to_district.csv')
feature_df = feature_df.merge(district_df, how = 'left', left_on = 'household_taz', right_on = 'TAZ')

luz_df = pd.read_excel('xref_taz_luz.xlsx')
luz_dict = dict(map(lambda i,j: (i,j), luz_df['taz'], luz_df['luz']))

feature_df['household_taz'] = feature_df['household_taz'].map(luz_dict)
feature_df['origin'] = feature_df['origin'].map(luz_dict)
feature_df['destination'] = feature_df['destination'].map(luz_dict)


feature_df.dropna(subset = ['origin', 'destination'], inplace = True)
district_list = [1,2,5,6]
feature_df = feature_df[feature_df['district'].isin(district_list)]


# In[7]:


X = feature_df[['trip_purpose','tour_type', 'tour_category', 'number_of_participants', 
                'start_time', 'age', 'student_status', 
                'employment_status', 'education', 
                'household_income', 'origin']]
y = feature_df['destination']


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


model = DecisionTreeClassifier(random_state = 42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)

logging.info("Baseline Accuracy on Train Set: {:.2f}%".format(train_accuracy * 100))
logging.info("Baseline Accuracy on Test Set: {:.2f}%".format(test_accuracy* 100))

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred, average = 'weighted')
recall = recall_score(y_test, y_pred, average = 'weighted')
f1 = f1_score(y_test, y_pred, average = 'weighted')

logging.info("Baseline Precision on Test Set: {:.2f}%".format(precision * 100))
logging.info("Baseline Recall on Test Set: {:.2f}%".format(recall * 100))
logging.info("Baseline F1 Score on Test Set: {:.2f}%".format(f1 * 100))


# In[10]:


max_depth = model.get_depth()
min_samples_split = model.min_samples_split
min_samples_leaf = model.min_samples_leaf
min_weight_fraction_leaf = model.min_weight_fraction_leaf
min_impurity_decrease = model.min_impurity_decrease

logging.info("Max Depth: {}, Min Split: {}, Min Leaf: {}, Min Weight: {}, Min Impurity: {}".format(max_depth, 
                                                                                                min_samples_split,
                                                                                                min_samples_leaf,
                                                                                                min_weight_fraction_leaf,
                                                                                                min_impurity_decrease
                                                                                               ))


# ## Oversampling

# In[11]:


# ## Class Imbalance - SMOTE

#sample_df = feature_df.groupby('destination', 
#                               group_keys=False).apply(lambda x:x.sample(frac=0.8))
X = feature_df[['trip_purpose','tour_type', 'tour_category', 'number_of_participants', 
                'start_time', 'age', 'student_status', 
                'employment_status', 'education', 
                'household_income', 'origin']]
y = feature_df['destination']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.2, random_state=42)


# In[12]:


num_samples = int(np.ceil(y_train.value_counts().median()))
sampling_strategy = {label: num_samples if count < num_samples else count for label, count in Counter(y_train).items()}

smote = SMOTE(k_neighbors = 4, sampling_strategy = sampling_strategy, random_state = 42)

X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


# In[13]:


model = DecisionTreeClassifier()

model.fit(X_resampled, y_resampled)

y_pred = model.predict(X_test)
y_train_pred = model.predict(X_resampled)

train_accuracy = accuracy_score(y_resampled, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)

logging.info("SMOTE Accuracy on Train Set: {:.2f}%".format(train_accuracy * 100))
logging.info("SMOTE Accuracy on Test Set: {:.2f}%".format(test_accuracy* 100))

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred, average = 'weighted')
recall = recall_score(y_test, y_pred, average = 'weighted')
f1 = f1_score(y_test, y_pred, average = 'weighted')

logging.info("SMOTE Precision on Test Set: {:.2f}%".format(precision * 100))
logging.info("SMOTE Recall on Test Set: {:.2f}%".format(recall * 100))
logging.info("SMOTE F1 Score on Test Set: {:.2f}%".format(f1 * 100))


# ## GridSearch

# In[16]:


# Define the hyperparameter grid

param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [45, 50, 55],  
    'min_samples_split': [2, 5, 10],   
    'min_samples_leaf': [1, 5, 10],     
    'min_weight_fraction_leaf': [0, 0.1, 0.5],  
    'min_impurity_decrease': [0, 0.1, 0.5]    
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state = 42), param_grid=param_grid, cv=5, scoring='f1_weighted')


# In[17]:


# Fit the GridSearchCV instance to the training data
try:
    grid_search.fit(X_resampled, y_resampled)
    logging.info("Grid Search Fit")
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    logging.info("Optimal Parameters: {}".format(best_params))
    logging.info("Optimal F1 Score: {}".format(best_score))
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_train_pred = best_model.predict(X_resampled)

    # Calculate accuracy, precision, recall, F1 score, R-Squared
    train_accuracy = accuracy_score(y_resampled, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average = 'weighted')
    recall = recall_score(y_test, y_pred, average = 'weighted')
    f1 = f1_score(y_test, y_pred, average = 'weighted')
    r2 = r2_score(y_test, y_pred)
    
    logging.info("SMOTE + GridSearch Accuracy on Train Set: {:.2f}%".format(train_accuracy * 100))
    logging.info("SMOTE + GridSearch Accuracy on Test Set: {:.2f}%".format(test_accuracy* 100))
    logging.info("SMOTE + GridSearch Precision on Test Set: {:.2f}%".format(precision * 100))
    logging.info("SMOTE + GridSearch Recall on Test Set: {:.2f}%".format(recall * 100))
    logging.info("SMOTE + GridSearch F1 Score on Test Set: {:.2f}%".format(f1 * 100))
    logging.info("SMOTE + GridSearch R2 Score on Test Set: {:.2f}".format(r2))
    
except Exception as e:
    logging.info("An unexpected error occurred: %s", e)

