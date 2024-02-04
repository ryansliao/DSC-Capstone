import sys
import json

sys.path.insert(0, 'src')

from etl import get_data
from features import apply_features
from model import predict_destination

def main(targets):

    if 'data' in targets:
        data = get_data()
        print("Data Read")

    if 'features' in targets:
        with open('config/features-params.json') as fh:
            feats_cfg = json.load(fh)
        features_df, model = apply_features(data, **feats_cfg)
        print("Data Pre-Processed")

    if 'model' in targets:
        predict_destination(features_df, model)
        print("Model Completed")

    if 'all' in targets:
        predict_destination(features_df, model)
        print("All Models Completed")

    return

if __name__=="__main__":

    targets = sys.argv[1:]
    main(targets)