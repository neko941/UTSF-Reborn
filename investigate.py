import glob
import polars as pl
import os
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(100)

# dfs = []
# for csv in glob.glob(r'.\runs\cases\*\*.csv'):
# 	df = pl.read_csv(csv)
# 	if int(csv.split(os.sep)[-2].replace('case', '')) > 18: continue
# 	df = df.with_columns(pl.lit(csv.split(os.sep)[-2].replace('case', '')).alias('case'))
# 	df = df.with_columns(pl.col('SMAPE').cast(pl.Utf8))
# 	dfs.append(df)
# dfs = pl.concat(dfs)

# print(dfs.sort(by='MAE')[0])
# print(dfs.sort(by='R2')[-1])
# print(dfs.filter(pl.col('Name') == 'ExtremeGradientBoostingRegression').sort(by='MAE', descending=True))
# dfs.filter(pl.col('Name') == 'ExtremeGradientBoostingRegression').sort(by='MAE', descending=True).write_csv('best_XGBoost.csv')
# print(dfs.filter(pl.col('Name') == 'build_Baseline_ave').sort(by='MAE', descending=True))

import json
import numpy as np
import itertools

def report():
	data = []
	a = [['R'], ['No', 'Yes', 'YesTrain'], [5, 15], [5, 12, 20]]
	data.extend(list(itertools.product(*a)))
	a = [['C'], ['No'], [5, 15], [5, 12, 20]]
	data.extend(list(itertools.product(*a)))
	# d = []
	b = []
	for i in glob.glob(r'.\runs\cases\*'):
		case = data[int(i.split(os.sep)[-1].replace('case', ''))]
		case = f'l{case[3]}-g{case[2]}-{case[1]}-{case[0]}'

		if os.path.exists(os.path.join(i, 'values\invalid.npy')): n = len(np.load(os.path.join(i, 'values\invalid.npy')))
		else: n = len(np.load(os.path.join(f'runs\cases\case{int(i.split(os.sep)[-1].replace("case", "")) - 6}', 'values\invalid.npy')))

		df = pl.read_csv(os.path.join(i, 'results.csv')).filter(pl.col('Name') == 'ExtremeGradientBoostingRegression')
		df = df.with_columns(pl.lit(case).alias('Des'))
		df = df.with_columns(pl.lit(int(i.split(os.sep)[-1].replace('case', ''))).alias('Case'))
		df = df.with_columns(pl.col('SMAPE').cast(pl.Float64))
		df = df.with_columns(pl.lit(n).alias('Num'))
		df = df.with_columns(pl.col('Time').str.split('m')).with_columns(pl.col('Time').arr.get(0) * 60 if pl.col('Time').arr.count() == 0 else pl.col('Time').arr.get(0))
		print(df)
		df = df.drop('Name')
		b.append(df)
	print(pl.concat(b).select(['Case', 'Des', 'Num', 'Time', 'MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE', 'R2']))
	pl.concat(b).select(['Case', 'Des', 'Num', 'Time', 'MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE', 'R2']).write_csv('best_XGBoost_2.csv')

def est_time():
	
	
est_time()