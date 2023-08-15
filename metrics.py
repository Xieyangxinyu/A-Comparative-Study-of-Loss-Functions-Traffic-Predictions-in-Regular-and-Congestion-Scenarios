import numpy as np
import pickle
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str, default="GraphWaveNet", help='enter path to the saved prediction file')
#parser.add_argument('--model',type=str, default="D2STGNN", help='enter path to the saved prediction file')
parser.add_argument('--dataset', default= "METR-LA",type=str, help='dataset')
#parser.add_argument('--dataset', default= "PEMS-BAY",type=str, help='dataset')

args = parser.parse_args()
congested = pickle.load(open(f"{args.dataset}/congested.pkl", "rb"))
cp_int_file = f"./{args.dataset}/change_point_intervals.pkl"
change_point_intervals = np.array(pickle.load(open(cp_int_file, "rb")))

model = 'wave'
if args.model == 'D2STGNN':
    model = 'D2STGNN'

if args.dataset == "PEMS-BAY":
    intervals = [0, 10, 20, 30, 40, 50, 60, 70, 80] # PEMS-BAY
else:
    intervals = [0, 10, 20, 30, 40, 50, 60] # METR-LA

pred_dir = Path(args.dataset) / args.model

# METR-LA
total_length = 34272
test_length = 6850
num_nodes = 207

if args.dataset == "PEMS-BAY":
    total_length = 52116
    test_length = 10419
    num_nodes = 325

horizons = [3, 6, 12]

def bMAE(mae, realy):
    mae_bin = []
    for interval in intervals:
        mae_bin.append(np.mean(mae[((realy > interval) & (realy <= interval + 10))]))
    return np.mean(mae_bin)

def bRMSE(mae, realy):
    mae_bin = []
    for interval in intervals:
        mae_bin.append(np.sqrt(np.mean(mae[((realy > interval) & (realy <= interval + 10))])))
    #print(mae_bin)
    return np.mean(mae_bin)

def cp_metric(change_points, pred_file):
    result = []
    for horizon in horizons:
        start = total_length - test_length - (12 - horizon)
        end = total_length - (12 - horizon)
        realy = pred_file[f"real{horizon}"]
        preds = pred_file[f"pred{horizon}"]

        mae = []
        mape = []
        mse = []

        #for i in range(num_nodes):
        for i in congested:
            change_point = change_points[i]
            change_point = np.array(change_point)
            change_point = change_point[((change_point > start) & (change_point < end))] - start

            y = preds[:, i][change_point]
            y_hat = realy[:, i][change_point]
            y = y[y_hat > 0]
            y_hat = y_hat[y_hat > 0]
            mae.append(np.mean(np.abs(y - y_hat)))
            mape.append(np.mean(np.abs(y - y_hat) / y))
            mse.append(np.mean((y - y_hat) ** 2))

        mae = np.mean(mae)
        
        mse = np.mean(mse)

        rmse = np.sqrt(mse)

        mape = np.mean(mape) * 100
        result = result + [mae, rmse, mape]
    return result


print(f"Metrics for : {args.model}-{args.dataset}")

def get_metrics(pred_file):
    common = []
    var_ae = []
    for horizon in horizons:
        realy = pred_file[f"real{horizon}"]
        preds = pred_file[f"pred{horizon}"]

        preds = preds[realy > 0]
        realy = realy[realy > 0]

        mae = np.abs(realy - preds)
        for percentile in [.95, .98, .99]:
            var_ae.append(np.quantile(mae, percentile))

        mape = mae / realy * 100
        mae = np.mean(mae)
        common.append(mae)


        mse = (realy - preds) ** 2
        mse = np.mean(mse)
        common.append(np.sqrt(mse))

        mape = np.mean(mape)
        common.append(mape)
    
    return common, cp_metric(change_point_intervals, pred_file), var_ae

loss_lookup = {'mae': 'MAE', 
               'mse': 'MSE', 
               'mae-focal': 'MAE-Focal', 
               'mse-focal': 'MSE-Focal', 
               'bmse1': 'bMSE-1', 
               'bmse9': 'bMSE-9', 
               'huber': 'Huber',
               'Gumbel': 'Gumbel',
               'kirtosis': 'Kirtosis',
               'quantile': 'Quantile'
               }

pd.set_option("display.precision", 3)

def add_line(df, loss, line):
    line = [f'{loss_lookup[loss]}'] + line
    df.loc[len(df)] = line

def highlight_min(s):
    second_smallest = s.nsmallest(2).values
    smallest = min(second_smallest)
    format_ = []
    for v in s:
        if v == smallest:
            format_.append('textbf:--rwrap;')
        elif v in second_smallest:
            format_.append('underline:--rwrap;')
        else:
            format_.append('')
    return format_

tables = ["Overall performance of loss functions with MAE, RMSE, MAPE", 
          "Performance of loss functions with MAE, RMSE, MAPE at identified congestion scenarios",
          "VaR of each loss function at three different levels: 95%, 98%, and 99%"]

for index in range(3):
    for losses in [['mae', 'mae-focal', 'quantile', 'huber', 'mse', 'mse-focal', 'bmse1', 'bmse9', 'Gumbel', 'kirtosis']]:
        dfs = [pd.DataFrame(columns=['loss', 'MAE-3', 'RMSE-3', 'MAPE-3', 'MAE-6', 'RMSE-6', 'MAPE-6', 'MAE-12', 'RMSE-12', 'MAPE-12']),
            pd.DataFrame(columns=['loss', 'MAE-3', 'RMSE-3', 'MAPE-3', 'MAE-6', 'RMSE-6', 'MAPE-6', 'MAE-12', 'RMSE-12', 'MAPE-12']),
            pd.DataFrame(columns=['loss', '95-3', '98-3', '99-3', '95-6', '98-6', '99-6', '95-12', '98-12', '99-12']),
        ]
        for loss in losses:
            pred_file = pickle.load(open(pred_dir / f'{model}-{loss}.pkl', "rb"))
            results = get_metrics(pred_file)
            for i in range(len(dfs)):
                add_line(dfs[i], loss, results[i])

        dfs[index] = dfs[index].set_index('loss')
        print(tables[index])
        print(dfs[index].style.apply(highlight_min, axis = 0).to_latex())

        
