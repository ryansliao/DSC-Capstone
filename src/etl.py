import numpy as np
import pandas as pd
import geopandas as gpd
import os
<<<<<<< HEAD
import re

def get_data(indir, outdir):

    # Read Files
    if not os.path.exists(indir):
        os.makedirs(indir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    households_dir = os.path.join(indir, 'households.csv')
    households = pd.read_csv(households_dir, low_memory=False)
    persons_dir = os.path.join(indir, 'persons.csv')
    persons = pd.read_csv(persons_dir, low_memory=False)
    trips_dir = os.path.join(indir, 'final_trips.csv')
    trips = pd.read_csv(trips_dir, low_memory=False)
    tours_dir = os.path.join(indir, 'final_tours.csv')
    tours = pd.read_csv(tours_dir, low_memory=False)
    zones_dir = os.path.join(indir, 'mgra15_based_input2022.csv')
    zones = pd.read_csv(zones_dir, low_memory=False)
    luz_map_dir = os.path.join(indir, 'LUZ')
    luz_map = gpd.read_file(luz_map_dir)
    taz_district_dir = os.path.join(indir, 'TAZ/taz_to_district.csv')
    taz_district = pd.read_csv(taz_district_dir)
    taz_luz_dir = os.path.join(indir, 'LUZ/xref_taz_luz.csv')
    taz_luz = pd.read_csv(taz_luz_dir)
    luz_distance_dir = os.path.join(indir, 'LUZ/auto_AM_2022.csv')
    luz_distance = pd.read_csv(luz_distance_dir)

    print("Data Read.")

    return households, persons, trips, tours, zones, luz_map, taz_district, taz_luz, luz_distance

def clean_data(households, persons, trips, tours, zones, luz_map, taz_district, taz_luz, luz_distance):

    # Cleaning Datasets
    households = households.rename(columns={'hhid': 'household_id'}).drop(['Unnamed: 0.4','Unnamed: 0.3','Unnamed: 0.2','Unnamed: 0.1', 'Unnamed: 0'], axis=1)
    persons = persons.rename(columns={'perid': 'person_id', 'hhid': 'household_id'}).drop(['Unnamed: 0.4','Unnamed: 0.3','Unnamed: 0.2','Unnamed: 0.1', 'Unnamed: 0'], axis=1)
    zones = zones.drop('Unnamed: 0', axis=1)

    feature_df = pd.merge(tours, persons, how='left', on=['person_id', 'household_id'])
    feature_df = pd.merge(feature_df, households[['household_id', 'taz', 'hinc']], how='left', on='household_id')
    feature_df = pd.merge(trips[['trip_id', 'tour_id', 'purpose']], feature_df, how='left', on='tour_id')
    feature_df = feature_df.drop(['tdd', 'tour_id_temp', 'destination_logsum', 'vehicle_occup_1', 'vehicle_occup_2', 'vehicle_occup_3.5', 'mode_choice_logsum'], axis=1)
    feature_df = feature_df.rename(columns={'purpose': 'trip_purpose', 'start': 'start_time', 'end': 'end_time', 'pemploy': 'employment_status', 'pstudent': 'student_status', 'ptype': 'person_type', 'educ': 'education', 'weeks': 'weeks_worked', 'hours': 'hours_worked', 'rac1p': 'race', 'taz': 'household_taz', 'hinc': 'household_income'})
    feature_df = feature_df.drop(['primary_purpose', 'household_serial_no', 'miltary', 'pnum', 'occen5', 'occsoc5', 'indcen', 'hisp', 'version', 'naics2_original_code', 'soc2'], axis=1)
    
    vehicles = feature_df['selected_vehicle'].str.split('_', expand=True)
    feature_df['vehicle_type'] = vehicles[0]
    feature_df['vehicle_age'] = vehicles[1]
    feature_df['vehicle_propulsion'] = vehicles[2]
    feature_df = feature_df.drop(['selected_vehicle'], axis=1)

    stops = feature_df['stop_frequency'].str.replace(r'[^0-9]', '', regex=True).str.split('', expand=True).drop([0, 3], axis=1)
    feature_df['out_stops'] = stops[1]
    feature_df['in_stops'] = stops[2]
    feature_df = feature_df.drop(['stop_frequency'], axis=1)

    zones_dict = zones[['mgra', 'taz']].to_dict('list')
    zones_dict_2 = dict(map(lambda i,j: (i,j), zones_dict['mgra'], zones_dict['taz']))
    feature_df['destination'] = feature_df['destination'].map(zones_dict_2)
    feature_df['origin'] = feature_df['origin'].map(zones_dict_2)

    trip_purpose = {'home': 1,
               'shopping': 2,
               'othmaint': 3,
               'work': 4,
               'othdiscr': 5,
               'eatout': 6,
               'escort': 7,
               'atwork': 8,
               'social': 9,
               'school': 10,
               'univ': 11}
    feature_df['trip_purpose'] = feature_df['trip_purpose'].map(trip_purpose)

    tour_category = {'non_mandatory': 1,
                'mandatory': 2,
                'joint': 3,
                'atwork': 4}
    feature_df['tour_category'] = feature_df['tour_category'].map(tour_category)

    tour_type = {'work': 1,
             'othdiscr': 2,
             'escort': 3,
             'othmaint': 4,
             'shopping': 5,
             'school': 6,
             'eatout': 7,
             'eat': 8,
             'social': 9,
             'business': 10,
             'maint': 11}
    feature_df['tour_type'] = feature_df['tour_type'].map(tour_type)

    feature_df['start_time'] = feature_df['start_time'] / 2
    feature_df['end_time'] = feature_df['end_time'] / 2
    feature_df['duration'] = feature_df['duration'] / 2

    vehicle_type = {'Car': 1,
               'SUV': 2,
               'Pickup': 3,
               'Van': 4,
               'Motorcycle': 5,
               'non': np.nan}
    feature_df['vehicle_type'] = feature_df['vehicle_type'].map(vehicle_type)

    vehicle_propulsion = {'Gas': 1,
                     'Hybrid': 2,
                     'veh': np.nan,
                     'Diesel': 3,
                     'PEV': 4,
                     'BEV': 5}
    feature_df['vehicle_propulsion'] = feature_df['vehicle_propulsion'].map(vehicle_propulsion)

    true_false = {True: 1,
              False: 2}
    feature_df['is_external_tour'] = feature_df['is_external_tour'].map(true_false)
    feature_df['is_internal_tour'] = feature_df['is_internal_tour'].map(true_false)
            
    # Merging Datasets
    taz_district = taz_district.rename(columns={'TAZ': 'household_taz'})
    feature_df = pd.merge(feature_df, taz_district, how='left', on=['household_taz'])
    luz_map = luz_map.rename(columns={'LUZ': 'household_luz'})
    luz_dict = taz_luz[['taz', 'luz']].to_dict('list')
    luz_dict_2 = dict(map(lambda i,j: (i,j), luz_dict['taz'], luz_dict['luz']))
    feature_df = feature_df.rename(columns={'household_taz': 'household_luz'})
    feature_df['household_luz'] = feature_df['household_luz'].map(luz_dict_2)
    feature_df['destination'] = feature_df['destination'].map(luz_dict_2)
    feature_df['origin'] = feature_df['origin'].map(luz_dict_2)

    feature_df = pd.merge(feature_df, luz_map, how='left', on=['household_luz'])
    feature_df = feature_df[feature_df['district'].isin([1, 2, 5, 6])]
    feature_df = feature_df.dropna(subset=['origin', 'destination'])

    print("Data Cleaned.")

    return feature_df, luz_map, luz_distance
=======

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
>>>>>>> 095cd869ceead124c900a0b08d5404db993cdb9e
