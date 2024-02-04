import pandas as pd
import geopandas as gpd

def get_data():
    # Reading Files
    features = pd.read_csv("input_data/feature_df.csv", low_memory=False)
    taz_map = gpd.read_file('input_data/TAZ')
    taz_district = pd.read_csv('input_data/TAZ/taz_to_district.csv')

    # Merging Datasets
    taz_map = taz_map.rename(columns={'TAZ': 'household_taz'})
    taz_district = taz_district.rename(columns={'TAZ': 'household_taz'})
    taz_map = pd.merge(taz_map, taz_district, how='left', on=['household_taz'])

    features = pd.merge(features, taz_map, how='left', on=['household_taz'])
    features = features[features['district'].isin([1, 2, 5, 6])]

    return features