import pandas as pd

def apply_features(feature_df, luz_map, luz_distance, inputs, model):
    return feature_df[inputs], feature_df[model], luz_map, luz_distance
