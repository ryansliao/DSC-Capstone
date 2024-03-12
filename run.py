import sys
import json

sys.path.insert(0, 'src')

from etl import get_data, clean_data
from features import apply_features
from model import predict_destination

def main(targets):

    if 'data' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)

<<<<<<< HEAD
        households, persons, trips, tours, zones, luz_map, taz_district, taz_luz, luz_distance = get_data(**data_cfg)
        feature_df, luz_map, luz_distance = clean_data(households, persons, trips, tours, zones, luz_map, taz_district, taz_luz, luz_distance)
=======
        features, luz_map, taz_district, taz_luz, luz_distance = get_data(**data_cfg)
>>>>>>> 095cd869ceead124c900a0b08d5404db993cdb9e

    if 'features' in targets:
        with open('config/features-params.json') as fh:
            feats_cfg = json.load(fh)

<<<<<<< HEAD
        feature_df, model, luz_map, luz_distance = apply_features(feature_df, luz_map, luz_distance, **feats_cfg)
=======
        features_df, model, luz_map, luz_distance = apply_features(features, luz_map, taz_district, taz_luz, luz_distance, **feats_cfg)
>>>>>>> 095cd869ceead124c900a0b08d5404db993cdb9e

    if 'model' in targets:
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)

<<<<<<< HEAD
        predict_destination(feature_df, model, luz_map, luz_distance, **model_cfg)
=======
        predict_destination(features_df, model, luz_map, luz_distance, **model_cfg)
>>>>>>> 095cd869ceead124c900a0b08d5404db993cdb9e

    return

if __name__=="__main__":

    targets = sys.argv[1:]
    main(targets)