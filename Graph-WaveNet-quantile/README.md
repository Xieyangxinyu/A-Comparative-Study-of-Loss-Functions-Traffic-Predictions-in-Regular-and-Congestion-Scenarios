# Graph WaveNet for Deep Spatial-Temporal Graph Modeling

This is the adapted pytorch implementation of Graph WaveNet in the following paper: 
[Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019] (https://arxiv.org/abs/1906.00121). We made changes to the code in order to adapt to the Quantile Loss implemented in [Quantifying Uncertainty in Deep Spatiotemporal Forecasting, KDD 2021] (https://dl.acm.org/doi/abs/10.1145/3447548.3467325).


<p align="center">
  <img width="350" height="400" src=./fig/model.png>
</p>

## Requirements
- python 3
- see `requirements.txt`


## Data Preparation

### Step1: Download METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN).

### Step2: Process raw data 

```
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

```
## Train Commands

```
# METR-LA
python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj  --save archive/quantile/

# PEMS-BAY
python train.py --data data/PEMS-BAY --gcn_bool --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --addaptadj  --randomadj --num_nodes 325 --save archive/quantile/
```


