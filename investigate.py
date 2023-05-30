import glob
import polars as pl
import os
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(100)

dfs = []
for csv in glob.glob(r'.\runs\cases\*\*.csv'):
	df = pl.read_csv(csv)
	if int(csv.split(os.sep)[-2].replace('case', '')) < 18: continue
	df = df.with_columns(pl.lit(csv.split(os.sep)[-2].replace('case', '')).alias('case'))
	df = df.with_columns(pl.col('SMAPE').cast(pl.Utf8))
	dfs.append(df)
	# print(df); exit()
	# if dfs is None: dfs = df
	# else: 
	# 	dfs = pl.concat([dfs, df])
dfs = pl.concat(dfs)

print(dfs.sort(by='MAE')[0])
print(dfs.sort(by='R2')[-1])
print(dfs.filter(pl.col('Name') == 'ExtremeGradientBoostingRegression').sort(by='MAE', descending=True))
print(dfs.filter(pl.col('Name') == 'build_Baseline_ave').sort(by='MAE', descending=True))