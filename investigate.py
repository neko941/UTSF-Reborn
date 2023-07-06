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
from utils.general import yaml_load

def report():
	data = []
	# a = [['R'], ['No', 'Yes', 'YesTrain'], [5, 15], [5, 12, 20]]
	# data.extend(list(itertools.product(*a)))
	# a = [['C'], ['No'], [5, 15], [5, 12, 20]]
	# data.extend(list(itertools.product(*a)))
	# d = []
	b = []
	for i in glob.glob(r'.\runs\*'):
		try:
			if not 'final' in i: continue
			if '.zip' in i: continue
			# if int(i.split(os.sep)[-1].replace('case', '')) < 47: continue
			# if int(i.split(os.sep)[-1].replace('case', '')) > 55: continue

			if os.path.exists(os.path.join(i, 'values\invalid.npy')): n = len(np.load(os.path.join(i, 'values\invalid.npy')))
			else: n = len(np.load(os.path.join(f'runs\cases\case{int(i.split(os.sep)[-1].replace("case", "")) - 6}', 'values\invalid.npy')))

			attributes = ['Case', 'Name', 'Lr', 'Epochs', 'Lag', 'Granularity', 'Time', 'MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE', 'R2', 'Num']

			df = pl.read_csv(os.path.join(i, 'results.csv')).filter(~pl.col("Name").str.contains("Baseline"))
			opt = yaml_load(f'{i}\\configs\\updated_opt.yaml')
			df = df.with_columns(pl.lit(int(i.split(os.sep)[-1].replace('case', '').replace('final', ''))).alias('Case'),
								 pl.col('SMAPE').cast(pl.Float64),
								 pl.lit(n).alias('Num'),
								 pl.lit([str(len(pl.read_csv(f'{i}\\logs\\{n}.csv'))) if os.path.exists(f'{i}\\logs\\{n}.csv') else '_' for n in df['Name'].to_list()]).alias('Epochs'),
								 pl.lit(opt['lag']).alias('Lag'),
								 pl.lit(opt['resample']).alias('Granularity'),
								 pl.lit(opt['lr']).alias('Lr'),
								 pl.lit(i).alias('Path'))
			if os.path.exists(f'{i}\\configs\\ExtremeGradientBoostingRegression.yaml'):
				opt = yaml_load(f'{i}\\configs\\ExtremeGradientBoostingRegression.yaml')
				df = df.with_columns(pl.lit(opt['eta']).alias('Lr'),
									 pl.lit(opt['tree_method']).alias('TreeMethod'),
									 pl.lit(opt['n_estimators']).alias('Epochs'))
				attributes.append('TreeMethod')
			df = df.with_columns(pl.lit(np.load(f'{i}\\values\\ytest.npy').mean()).alias('ymean'),
								 pl.format("{}\\values\\yhat-{}.npy", pl.col('Path'), pl.col('Name')).alias('yhatmeanpath'))
			df = df.with_columns(pl.col('yhatmeanpath').apply(lambda path: np.load(str(path)).mean()).alias('yhatmean'))
			df.drop('yhatmeanpath')
			attributes.extend(['ymean', 'yhatmean'])


			# df = df.with_columns(pl.format("{}-lr{}-l{}-g{}", pl.col('Name'), pl.col('Lr'), pl.col('Lag'), pl.col('Granularity')).alias('CaseName'))
			# df = df.with_columns(pl.col('CaseName').str.replace('ExtremeGradientBoostingRegression', 'XGBoost'))
			# attributes.append('CaseName')
			b.append(df)
		except:
			pass
	# attributes = list(set(attributes))
	# print(b)
	df = pl.concat(b).select(attributes)
	df = df.sort('Case')
	print(df)
	df.write_csv('XGBoost.csv')

def est_time():
	est = []
	for path in glob.glob(r'.\runs\cases\*'):
		if '.zip' in path: continue
		if int(path.split(os.sep)[-1].replace('case', '')) < 47: continue
		if int(path.split(os.sep)[-1].replace('case', '')) > 55: continue
		samples = len(np.load(os.path.join(path, r'values\ytrain.npy')))

		def convert_to_seconds(time_str):
		    if 'm' in time_str:
		        minutes, seconds = time_str.split('m')
		        minutes = int(minutes)
		        seconds = int(seconds.rstrip('s'))
		        total_seconds = minutes * 60 + seconds
		    else:
		        total_seconds = int(time_str.rstrip('s'))
		    return total_seconds


		time = convert_to_seconds(pl.read_csv(os.path.join(path, r'results.csv')).filter(pl.col('Name') == 'ExtremeGradientBoostingRegression')['Time'].to_list()[0])

		est.append(time / samples)

	print(est)
	print(np.array(est).mean())

def report_updated(path):
	data = pl.DataFrame()
	for i in glob.glob(f'{path}\\*\\*'):
		df = pl.read_csv(os.path.join(i, 'results.csv'), 
						 new_columns=['Name', 'Time', 'MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE', 'R2'],
						 has_header=False)
		df = df.with_columns(pl.col('Name').str.replace('ExtremeGradientBoostingRegression', 'XGBoost'))
		opt = yaml_load(f'{i}\\configs\\updated_opt.yaml')
		df = df.with_columns(pl.lit(opt['lag']).alias('lag'))
		df = df.with_columns(pl.lit(opt['resample']).alias('resample'))
		df = df.with_columns(pl.lit(i.split(os.sep)[-2]).alias('Case'))
		opt = yaml_load([f for f in glob.glob(f"{i}\\configs\\*.yaml") if 'traffic' in f][0])

		# df = df.with_columns(pl.lit('Time').cast(pl.Utf8))
		# print(df)

		n = len(np.load(os.path.join(i, r'values\invalid.npy')))
		df = df.with_columns(pl.lit(n).alias('invalid'))

		options = ['150_random_ids', 'all_ids']
		df = df.with_columns(pl.when(pl.col("Case").str.contains(options[1])).then(options[1]).otherwise(options[0]).alias('ids'))

		used_features = [opt['target'], *opt['features']]
		for f in ['kml_segment_id', 'avg_speed', 'min_speed', 'max_speed', 'std_speed', 'rain']:
			df = df.with_columns(pl.lit(f in used_features).alias(f))
		data = pl.concat([data, df])
	print(data)
	data = data.sort(['R2'], descending=True)
	data.write_csv('all.csv')

report_updated(r'.\runs\20230628-avg_min_max_std_rain') # 