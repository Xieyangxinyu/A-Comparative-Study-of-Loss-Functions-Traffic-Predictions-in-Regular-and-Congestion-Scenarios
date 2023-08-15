# Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting

This is the adapted pytorch implementation of D2STGNN: "[Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting](https://arxiv.org/abs/2206.09112)". 

## 1. Table of Contents

```text
configs         ->  training Configs and model configs for each dataset
dataloader      ->  pytorch dataloader
datasets        ->  raw data and processed data
model           ->  model implementation and training pipeline
result          ->  predicted values at different horizons
archive         ->  model checkpoint
```

## 2. Requirements

```bash
pip install -r requirements.txt
```

## 3. Data Preparation

### 3.1 Download Data

For convenience, we package these datasets used in our model in [Google Drive](https://drive.google.com/drive/folders/1H3nl0eRCVl5jszHPesIPoPu1ODhFMSub?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1iFcKJ8qeCthyEgPEXYJ-rA?pwd=8888).

They should be downloaded to the code root dir and replace the `raw_data` and `sensor_graph` folder in the `datasets` folder by:

```bash
cd /path/to/project
unzip raw_data.zip -d ./datasets/
unzip sensor_graph.zip -d ./datasets/
rm {sensor_graph.zip,raw_data.zip}
mkdir log output
```

Alterbatively, the datasets can be found as follows:

- METR-LA and PEMS-BAY: These datasets were released by DCRNN[1]. Data can be found in its [GitHub repository](https://github.com/chnsh/DCRNN_PyTorch), where the sensor graphs are also provided.

### 3.2 Data Process

```bash
python datasets/raw_data/$DATASET_NAME/generate_training_data.py
```

Replace `$DATASET_NAME` with one of `METR-LA`, `PEMS-BAY`.

The processed data is placed in `datasets/$DATASET_NAME`.

## 4. Training the D2STGNN Model

```bash
python main.py --dataset=$DATASET_NAME --loss $LOSS
```
Replace `$LOSS` with one of `mae`, `mse`, `mae-focal`, `mse-focal`, `bmse1`, `bmse9`, `huber`, `kirtosis`, `Gumbel`.
E.g., `python main.py --dataset=METR-LA --loss mae`.