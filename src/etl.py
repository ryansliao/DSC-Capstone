import pandas as pd
import geopandas as gpd
import os

def read_features(datadir):
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    fp = os.path.join(datadir, 'feature_df.csv')
    return pd.read_csv(fp, low_memory=False)

def read_luz(datadir):
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    fp = os.path.join(datadir, 'LUZ')
    return gpd.read_file(fp)

def read_taz_district(datadir):
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    fp = os.path.join(datadir, 'TAZ/taz_to_district.csv')
    return pd.read_csv(fp)

def read_taz_luz(datadir):
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    fp = os.path.join(datadir, 'LUZ/xref_taz_luz.csv')
    return pd.read_csv(fp)

def read_luz_distance(datadir):
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    fp = os.path.join(datadir, 'LUZ/auto_AM_2022.csv')
    return pd.read_csv(fp)

def get_data(indir, outdir):
    features = read_features(indir)
    luz_map = read_luz(indir)
    taz_district = read_taz_district(indir)
    taz_luz = read_taz_luz(indir)
    luz_distance = read_luz_distance(indir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return features, luz_map, taz_district, taz_luz, luz_distance