import pandas as pd

def get_data():
    # Reading Files
    vehicles = pd.read_csv('input_data/vehpub_2017.csv')
    households = pd.read_csv('input_data/hhpub_2017.csv')

    # Merging Datasets
    vehicles = vehicles.merge(households[['HOUSEID', 'CNTTDHH',  'CAR', 'PRICE', 'PLACE']], on='HOUSEID', how='left')
    vehicles['VEHTYPE'] = vehicles['VEHTYPE'].replace(5, 4)

    return vehicles
