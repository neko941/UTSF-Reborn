# Ultimate Time Series Forcasting

# Installation

### Colab

### Window
```
python -m virtualenv venv --python=3.10.10
.\venv\Scripts\activate
pip install -r .\dependencies\requirements.txt
```

### Linux

# Usage
```
python .\train.py --lag=3 --ahead=1 --granularity=1440 --splitFeature=station --dataConfigs=.\configs\datasets\salinity-4_ids-split_column.yaml
```
```
python .\train.py --lag=3 --ahead=1 --dataConfigs=.\configs\datasets\traffic-1_id-split_column.yaml --delimiter='|' --granularity=5 --startTimeId=240
```