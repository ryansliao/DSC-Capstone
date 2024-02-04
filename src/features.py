import pandas as pd

def apply_features(df, inputs, model):
    df = df[['trip_purpose',
            'tour_type',
            'tour_category',
            'origin',
            'number_of_participants',
            'start_time',
            'age',
            'employment_status',
            'student_status',
            'education',
            'household_income']]
    return df[inputs], df[model]