import os
import time
import numpy as np
import polars as pl
from xgboost import XGBRegressor
from utils.general import yaml_load
from utils.general import convert_seconds
from utils.metrics import score
from utils.metrics import metric_dict
from itertools import product
from utils.rich2polars import table_to_df

from rich import box as rbox
from rich.table import Table
from rich.console import Console
from rich.terminal_theme import MONOKAI

save_dir = '.'
modelConfigs = r'.\runs\20230628-avg_min_max_std_rain\crossvalidation-data_avg_min_max_std_rain-all_ids-use_avg_min_max_std_rain\r15l20\configs\ExtremeGradientBoostingRegression.yaml'
xtrainpath = r'.\runs\20230628-avg_min_max_std_rain\crossvalidation-data_avg_min_max_std_rain-all_ids-use_avg_min_max_std_rain\r15l20\values\xtrain.npy'
ytrainpath = r'.\runs\20230628-avg_min_max_std_rain\crossvalidation-data_avg_min_max_std_rain-all_ids-use_avg_min_max_std_rain\r15l20\values\ytrain.npy'
xtestpath = r'.\runs\20230628-avg_min_max_std_rain\crossvalidation-data_avg_min_max_std_rain-all_ids-use_avg_min_max_std_rain\r15l20\values\xtest.npy'
ytestpath = r'.\runs\20230628-avg_min_max_std_rain\crossvalidation-data_avg_min_max_std_rain-all_ids-use_avg_min_max_std_rain\r15l20\values\ytest.npy'

def progress_bar():
	from rich.progress import Progress
	from rich.progress import BarColumn 
	from rich.progress import TextColumn
	from rich.progress import TimeElapsedColumn
	from rich.progress import MofNCompleteColumn
	from rich.progress import TimeRemainingColumn
	return Progress("[bright_cyan][progress.description]{task.description}",
					BarColumn(),
					TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
					TextColumn("•Items"),
					MofNCompleteColumn(), # "{task.completed}/{task.total}",
					TextColumn("•Remaining"),
					TimeRemainingColumn(),
					TextColumn("•Total"),
					TimeElapsedColumn())

def preprocessing(x, classifier=False):
		if classifier: res = [i.flatten().astype(int) for i in x]
		else: res = [i.flatten() for i in x]
		return res

xtr = preprocessing(np.load(file=xtrainpath, mmap_mode='r'))
ytr = np.ravel(preprocessing(np.load(file=ytrainpath, mmap_mode='r')))
xt = preprocessing(np.load(file=xtestpath, mmap_mode='r'))
yt = preprocessing(np.load(file=ytestpath, mmap_mode='r'))

params = {
	'n_estimators': [2000, 500, 1000, 5000],
	'max_depth': [2, 4, 6, 8, 10, 12, 14],
	'gamma': [0, 0.5, 1, 1.5, 2],
	'eta': [0.1, 0.01, 0.001], 
}

console = Console(record=True)
table = Table(title="[cyan]Results", show_header=True, header_style="bold magenta", box=rbox.ROUNDED, show_lines=True)
[table.add_column(f'[green]{name}', justify='center') for name in ['Name', 'Time', *list(params.keys()), *list(metric_dict.keys())]]

with progress_bar() as progress:
	for comb in progress.track([dict(zip(params.keys(), values)) for values in product(*params.values())]):
		print(comb)
		start = time.time()
		model = XGBRegressor(**yaml_load(modelConfigs))
		model.set_params(**comb)
		model.fit(xtr, ytr)
		yhat = model.predict(xt)
		table.add_row('XGBoost', convert_seconds(time.time() - start), *[str(v) for v in comb.values()], *score(y=yt, yhat=yhat, r=-1))
		console.print(table)
		console.save_svg(os.path.join(save_dir, 'results.svg'), theme=MONOKAI)  
		table_to_df(table).write_csv(os.path.join(save_dir, 'results.csv'))

# import polars as pl
# data = pl.read_csv('results.csv')
# data = data.with_columns(pl.lit(5).alias('Resample'),
# 						 pl.lit(20).alias('Lag'))
# data1 = pl.read_csv(r'.\runs\finetune\finetune-r5l5.csv')
# data1 = data1.with_columns(pl.lit(5).alias('Resample'),
# 						   pl.lit(5).alias('Lag'))
# pl.concat([data, data1]).write_csv('finetune.csv')