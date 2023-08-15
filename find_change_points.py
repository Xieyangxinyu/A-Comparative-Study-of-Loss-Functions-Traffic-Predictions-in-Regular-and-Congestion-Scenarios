import numpy as np
import pandas as pd
import ruptures as rpt
import pickle
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--path', default = 'METR-LA', type=str, help='enter path of the dataset')
#parser.add_argument('--path', default = 'PEMS-BAY', type=str, help='enter path of the dataset')
args = parser.parse_args()

path = Path(args.path)

if path == 'PEMS-BAY':
    ts_file = 'pems-bay.h5'
else:
    ts_file = 'metr-la.h5'

def get_change_points(tsd):
    tsd = np.array(tsd)
    detector = rpt.Pelt(model="rbf").fit(tsd)
    change_points = detector.predict(pen=3)
    return change_points[:-1]

def get_change_point_intervals(change_points, n):
    change_points = np.array(change_points)
    arrays = [change_points + i for i in range(-2,3)]
    change_point_intervals = np.unique(arrays)
    change_point_intervals = change_point_intervals[change_point_intervals > 0]
    change_point_intervals = change_point_intervals[change_point_intervals < n]
    return change_point_intervals

with pd.HDFStore(path / ts_file) as store:
    df = store[store.keys()[0]]

df.index = pd.to_datetime(df.index)
df = df.replace(to_replace=0.0, method='ffill')
n = len(df)

change_points = df.apply(get_change_points)
pickle.dump(change_points, open(path/"change_points.pkl", "wb"))

change_point_intervals = change_points.apply(get_change_point_intervals, n = n)
pickle.dump(change_point_intervals, open(path/"change_point_intervals.pkl", "wb"))