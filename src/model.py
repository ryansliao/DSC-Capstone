import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import seaborn as sns
from scipy import stats
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE

def predict_destination(features, labels, luz_map, luz_distance, classification, prediction_map, trip_distances, test_size, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_depth, criterion):

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, stratify=labels)

    # Adjust for class imbalance
    num_samples = int(np.ceil(y_train.value_counts().median()))
    sampling_strategy = {label: num_samples if count < num_samples else count for label, count in Counter(y_train).items()}
    smote = SMOTE(k_neighbors = 4, sampling_strategy = sampling_strategy)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Decision Tree Model
    st = time.time()
    model = DecisionTreeClassifier(min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth, criterion=criterion)
    model.fit(X_resampled, y_resampled)
    y_test_pred = model.predict(X_test)

    # Time model performance
    et = time.time()
    elapsed_time = et - st
    print('Model Execution Time:', elapsed_time, 'seconds')

    # Output classification report
    report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=np.nan)
    report_df = pd.DataFrame(report).transpose()[-3:]
    report_df.to_csv(classification)

    # Output prediction map
    results = pd.DataFrame(np.transpose(np.unique(y_test_pred, return_counts=True)), columns=['destination', 'count'])
    merged_res = luz_map.merge(results, how='left', left_on='household_luz', right_on='destination')
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    merged_res.plot(column='count', ax=ax, legend=True, cmap='viridis', linewidth=0.25, norm=matplotlib.colors.LogNorm(vmin=merged_res['count'].min(), vmax=merged_res['count'].max()))
    ax.set_title("Destination Prediction Map", fontsize=20)
    ax.set_axis_off()
    plt.savefig(prediction_map)

    # Output distance distribution plot
    luz_distance = luz_distance.rename(columns={'orig': 'origin'})
    luz_distance = luz_distance.rename(columns={'dest': 'destination'})

    pred_distance_df = pd.DataFrame(y_test_pred, columns=['destination']).reset_index().drop(columns=['index'])
    pred_distance_df[['origin', 'employment_status']] = pd.DataFrame(X_test[['origin', 'employment_status']]).reset_index()[['origin', 'employment_status']]

    real_distance_df = pd.DataFrame(y_test, columns=['destination']).reset_index().drop(columns=['index'])
    real_distance_df[['origin', 'employment_status']] = pd.DataFrame(X_test[['origin', 'employment_status']]).reset_index()[['origin', 'employment_status']]

    pred_distance_df = pd.merge(pred_distance_df, luz_distance[['origin', 'destination', 'distance']], on=['origin', 'destination'], how='left')
    pred_distance_df['Data Source'] = 'Our Model'

    real_distance_df = pd.merge(real_distance_df, luz_distance[['origin', 'destination', 'distance']], on=['origin', 'destination'], how='left')
    real_distance_df['Data Source'] = 'SANDAG Synthetic'

    merged = pd.concat([pred_distance_df, real_distance_df], axis=0)
    merged_plot = merged[merged['distance'] < 30].rename(columns={'employment_status': 'Employment Status'})
    merged_sample = merged_plot.sample(10000, replace=False)

    sns.set_theme()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax = sns.kdeplot(ax=ax, data=merged_sample, x='distance', alpha=1, hue='Data Source', linewidth=1.5, palette='tab10')
    plt.setp(ax.get_legend().get_title(), fontsize='16')
    plt.setp(ax.get_legend().get_texts(), fontsize='12')
    ax.set_title("Trip Destination Distance Distribution", fontsize=20)
    ax.set_xlabel("Trip Distance (miles)", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    plt.savefig(trip_distances)

    # Output distance comparison metric
    n_iterations = 5000
    bootstrap_stats = []
    observed_statistic = stats.ks_2samp(pred_distance_df['distance'], real_distance_df['distance']).statistic

    for _ in range(n_iterations):
        sample1 = pred_distance_df['distance'].sample(10000, replace=True)
        sample2 = real_distance_df['distance'].sample(10000, replace=True)
        bootstrap_stat = stats.ks_2samp(sample1, sample2).statistic
        bootstrap_stats.append(bootstrap_stat)

    p_value = np.sum(np.abs(bootstrap_stats) >= np.abs(observed_statistic)) / n_iterations

    print("2 Sample Kolmogorov-Smirnov Test Score:", observed_statistic)
    print("P-Value:", p_value)

    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis.")
    else:
        print("Failed to reject the null hypothesis.")