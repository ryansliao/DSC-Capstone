import pandas as pd
import geopandas as gpd

def get_data():
    # Reading Files
    features = pd.read_csv("input_data/feature_df.csv", low_memory=False)
    luz_map = gpd.read_file('input_data/LUZ')
    taz_district = pd.read_csv('input_data/TAZ/taz_to_district.csv')
    taz_luz = pd.read_csv('input_data/LUZ/xref_taz_luz.csv')
    
    # Merging Datasets
    taz_district = taz_district.rename(columns={'TAZ': 'household_taz'})
    features = pd.merge(features, taz_district, how='left', on=['household_taz'])
    luz_map = luz_map.rename(columns={'LUZ': 'household_luz'})
    luz_dict = taz_luz[['taz', 'luz']].to_dict('list')
    luz_dict_2 = dict(map(lambda i,j: (i,j), luz_dict['taz'], luz_dict['luz']))
    features = features.rename(columns={'household_taz': 'household_luz'})
    features['household_luz'] = features['household_luz'].map(luz_dict_2)
    features['destination'] = features['destination'].map(luz_dict_2)
    features['origin'] = features['origin'].map(luz_dict_2)

    features = pd.merge(features, luz_map, how='left', on=['household_luz'])
    features = features[features['district'].isin([1, 2, 5, 6])]
    features = features.dropna(subset=['origin', 'destination'])

    return features
