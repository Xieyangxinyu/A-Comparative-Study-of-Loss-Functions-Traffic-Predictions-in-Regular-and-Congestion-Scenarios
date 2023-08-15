import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import argrelextrema
import geopandas as gpd
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--path', default = 'METR-LA', type=str, help='enter path of the dataset')
args = parser.parse_args()
alpha = 0.1

data_path = args.path

if data_path == 'PEMS-BAY':
    FILE = f'{data_path}/pems-bay.h5'
    locations_file = f'{data_path}/graph_sensor_locations_bay.csv'
else:
    FILE = f'{data_path}/metr-la.h5'
    locations_file = f'{data_path}/graph_sensor_locations.csv'
    

with pd.HDFStore(FILE) as store:
    print(store.keys())
    df = store[store.keys()[0]]
    
df.replace(0, np.nan, inplace=True)

def is_bimodal(data):
    kde_values = sns.kdeplot(data, bw_adjust = 2).get_lines()[0].get_data()
    mode = kde_values[0][np.argmax(kde_values[1])]
    local_minima_indices = argrelextrema(kde_values[1], np.less)
    valley = kde_values[0][local_minima_indices]
    proportion = 0
    for v in valley:
        thresh = v
        if thresh < mode - 10:
            prop = sum(data < thresh)/len(data.dropna())
            if abs(prop - 0.5) < abs(proportion - 0.5):
                proportion = prop
    return min(proportion, 1 - proportion)

result = df.apply(is_bimodal, alpha)

if locations_file == f'{data_path}/graph_sensor_locations_bay.csv':
    locations = pd.read_csv(locations_file, header=None)
    locations.columns = ['sensor_id', 'latitude', 'longitude']
    locations = locations.set_index('sensor_id')
else:
    locations = pd.read_csv(locations_file, index_col=1)
locations.index = locations.index.astype(str)
result = pd.DataFrame(result, columns=['Proportion'])
result.index = result.index.astype(str)
merged = pd.concat([locations, result], axis=1)

gdf = gpd.GeoDataFrame(
    merged, geometry=gpd.points_from_xy(merged.longitude, merged.latitude), crs="EPSG:4326"
)

print(f"Number of bimodal sensors: {sum(result['Proportion'] > alpha)}, percentage: {sum(result['Proportion'] > 0.1)/len(result['Proportion'])}")

df.columns = df.columns.astype(str)
congested = [df.columns.get_loc(c) for c in result[result['Proportion'] > alpha].index if c in df]
pickle.dump(congested, open(f"{data_path}/congested.pkl", "wb"))
