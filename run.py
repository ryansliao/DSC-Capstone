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

        households, persons, trips, tours, zones, luz_map, taz_district, taz_luz, luz_distance = get_data(**data_cfg)
        feature_df, luz_map, luz_distance = clean_data(households, persons, trips, tours, zones, luz_map, taz_district, taz_luz, luz_distance)

    if 'features' in targets:
        with open('config/features-params.json') as fh:
            feats_cfg = json.load(fh)

        feature_df, model, luz_map, luz_distance = apply_features(feature_df, luz_map, luz_distance, **feats_cfg)

    if 'model' in targets:
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)

        predict_destination(feature_df, model, luz_map, luz_distance, **model_cfg)

    return

if __name__=="__main__":

    targets = sys.argv[1:]
    main(targets)