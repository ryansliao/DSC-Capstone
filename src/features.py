import pandas as pd

def apply_features(df):
    df = df[(df['VEHTYPE'] > 0)
            & (df['VEHTYPE'] != 6)
            & (df['VEHTYPE'] != 97)
            & (df['FUELTYPE'] > 0)
            & (df['FUELTYPE'] != 97)
            & (df['FUELTYPE'] > 0)
            & (df['FUELTYPE'] != 97)
            & (df['VEHAGE'] < 40)
            & (df['VEHAGE'] > 0)
            & (df['CAR'] > 0)
            & (df['HHFAMINC'] > 0)
            & (df['GSCOST'] != -9)
            & (df['PRICE'] > 0)
            & (df['PLACE'] > 0)
            & (df['URBAN'] != 4)]
    df = df.astype({'VEHTYPE': object})
    df = df.astype({'FUELTYPE': object})
    df.loc[df['VEHTYPE'] == 1, 'VEHTYPE'] = 'Automobile'
    df.loc[df['VEHTYPE'] == 2, 'VEHTYPE'] = 'Van'
    df.loc[df['VEHTYPE'] == 3, 'VEHTYPE'] = 'SUV'
    df.loc[df['VEHTYPE'] == 4, 'VEHTYPE'] = 'Truck'
    df.loc[df['VEHTYPE'] == 7, 'VEHTYPE'] = 'Motorcycle'
    df.loc[df['FUELTYPE'] == 1, 'FUELTYPE'] = 'Gas'
    df.loc[df['FUELTYPE'] == 2, 'FUELTYPE'] = 'Diesel'
    df.loc[df['FUELTYPE'] == 3, 'FUELTYPE'] = 'Hybrid/Electric'
    return df
