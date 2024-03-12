import pandas as pd

def apply_features(features, luz_map, taz_district, taz_luz, luz_distance, inputs, model):
    
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

    return features[inputs], features[model], luz_map, luz_distance