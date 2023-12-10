import sys
import json

sys.path.insert(0, 'src')

from etl import get_data
from features import apply_features
from vehtype_model import predict_vehicle_type
from fueltype_model import predict_vehicle_fuel
from vehage_model import predict_vehicle_age

def main(targets):

    if 'data' in targets:
        data = get_data()
        print("Data Read")

    if 'features' in targets:
        with open('config/features-params.json') as fh:
            feats_cfg = json.load(fh)
        features_df, vehtype, fueltype, vehage = apply_features(data, **feats_cfg)
        print("Data Pre-Processed")

    if 'vehtype' in targets:
        predict_vehicle_type(features_df, vehtype)
        print("Vehicle Type Model Completed")
    
    if 'fueltype' in targets:
        predict_vehicle_fuel(features_df, fueltype)
        print("Vehicle Fuel Type Model Completed")
    
    if 'vehage' in targets:
        predict_vehicle_age(features_df, vehage)
        print("Vehicle Age Model Completed")

    if 'all' in targets:
        predict_vehicle_type(features_df, vehtype)
        predict_vehicle_fuel(features_df, fueltype)
        predict_vehicle_age(features_df, vehage)
        
        print("All Models Completed")

    return

if __name__=="__main__":

    targets = sys.argv[1:]
    main(targets)
