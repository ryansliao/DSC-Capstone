import sys

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
        inputs = ['CAR', 'CNTTDHH', 'CDIVMSAR', 'DRVRCNT', 'GSCOST', 'HTPPOPDN', 'HHSIZE', 'HHFAMINC', 'HHVEHCNT', 'PLACE', 'PRICE', 'URBAN']
        features_df = apply_features(data)
        print("Data Pre-Processed")

    if 'vehtype' in targets:
        predict_vehicle_type(features_df[inputs], features_df['VEHTYPE'])
        print("Vehicle Type Model Completed")
    
    if 'fueltype' in targets:
        predict_vehicle_fuel(features_df[inputs], features_df['FUELTYPE'])
        print("Vehicle Fuel Type Model Completed")
    
    if 'vehage' in targets:
        predict_vehicle_age(features_df[inputs], features_df['VEHAGE'])
        print("Vehicle Age Model Completed")

    if 'all' in targets:
        predict_vehicle_type(features_df[inputs], features_df['VEHTYPE'])
        predict_vehicle_fuel(features_df[inputs], features_df['FUELTYPE'])
        predict_vehicle_age(features_df[inputs], features_df['VEHAGE'])
        print("All Models Completed")

    return

if __name__=="__main__":

    targets = sys.argv[1:]
    main(targets)
