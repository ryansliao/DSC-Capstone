import pandas as pd

def apply_features(df, inputs, model):
    return df[inputs], df[model]