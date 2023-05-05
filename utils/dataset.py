import os
import json
import time
import numpy as np
import polars as pl
from datetime import timedelta
from datetime import datetime
from dateutil.parser import parse
from multiprocessing.pool import ThreadPool

from utils.general import yaml_load
from utils.general import list_convert
from utils.general import flatten_list

from utils.option import model_dict

from rich.progress import track
from rich.progress import Progress
from rich.progress import BarColumn 
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import MofNCompleteColumn
from rich.progress import TimeRemainingColumn

class DatasetController():
    def __init__(self, 
                 configsPath=None, 
                 # granularity=1, 
                 # startTimeId=0, 
                 workers=8, 
                 splitRatio=(0.7, 0.2, 0.1), 
                 lag=5, 
                 ahead=1, 
                 offset=1, 
                 savePath='.', 
                 polarsFilling=None, 
                 machineFilling=None):
        """ Read data config """
        self.dataConfigs = yaml_load(configsPath)

        try:
            self.dataPaths = self.dataConfigs['data']
            self.dateFeature = self.dataConfigs['date']
            self.timeID = self.dataConfigs['time_id']
            self.targetFeatures = self.dataConfigs['target']
            self.delimiter = self.dataConfigs['delimiter']
            self.trainFeatures = list_convert(self.dataConfigs['features'])
            self.dirAsFeature = self.dataConfigs['dir_as_feature']
            self.splitDirFeature = self.dataConfigs['split_dir_feature']
            self.splitFeature = self.dataConfigs['split_feature']
            self.timeFormat = self.dataConfigs['time_format']
            self.granularity = self.dataConfigs['granularity']
            self.startTimeId = self.dataConfigs['start_time_id']

            self.yearStart   = self.dataConfigs['year_start']
            self.yearEnd     = self.dataConfigs['year_end']
            self.monthStart  = self.dataConfigs['month_start']
            self.monthEnd    = self.dataConfigs['month_end']
            self.dayStart    = self.dataConfigs['day_start']
            self.dayEnd      = self.dataConfigs['day_end']
            self.hourStart   = self.dataConfigs['hour_start']
            self.hourEnd     = self.dataConfigs['hour_end']
            self.minuteStart = self.dataConfigs['minute_start']
            self.minuteEnd   = self.dataConfigs['minute_end']
            
            self.X_train = []
            self.y_train = []
            self.X_val = []
            self.y_val = []
            self.X_test = []
            self.y_test = []
        except KeyError:
            self.X_train = np.load(self.dataConfigs['X_train'])
            self.y_train = np.load(self.dataConfigs['y_train'])
            self.X_val = np.load(self.dataConfigs['X_val'])
            self.y_val = np.load(self.dataConfigs['y_val'])
            self.X_test = np.load(self.dataConfigs['X_test'])
            self.y_test = np.load(self.dataConfigs['y_test'])

        self.configsPath = configsPath
        self.workers = workers
        # self.granularity = granularity
        # self.startTimeId = startTimeId
        self.splitRatio = splitRatio
        self.lag = lag
        self.ahead = ahead
        self.offset = offset
        self.savePath = savePath
        self.polarsFilling = polarsFilling
        self.machineFilling = machineFilling

        self.df = None
        self.dataFilePaths = []
        self.dirFeatures = []
        self.segmentFeature = None

        self.num_samples = []

    def execute(self, cyclicalPattern=False):
        if len(self.y_train) == 0:
            self.GetDataPaths(self.dataPaths)
            self.ReadFileAddFetures(csvs=self.dataFilePaths, dirAsFeature=self.dirAsFeature, hasHeader=True)
            self.df = self.df.drop_nulls()
            if self.timeFormat is not None: self.UpdateDateColumnDataType(dateFormat=self.timeFormat, f=pl.Datetime, t=pl.Datetime)
            self.TimeIDToDateTime(timeIDColumn=self.timeID, granularity=self.granularity, startTimeId=self.startTimeId)
            self.df = self.StripDataset(df=self.df)
            self.GetSegmentFeature(dirAsFeature=self.dirAsFeature, splitDirFeature=self.splitDirFeature, splitFeature=self.splitFeature)
            self.df = self.GetUsedColumn(df=self.df)
            if self.machineFilling: self.MachineLearningFillingMissingData(model=self.machineFilling)
            if self.polarsFilling: self.PolarsFillingMissingData(strategy=self.polarsFilling)
            if cyclicalPattern: self.CyclicalPattern()
            self.df = self.StripDataset(df=self.df)

            for i in range(10):
                self.df = self.df.with_columns(pl.col(self.targetFeatures).alias(f'thecol{i}'))
                self.trainFeatures.append(f'thecol{i}')

            self.SplittingData(splitRatio=self.splitRatio, lag=self.lag, ahead=self.ahead, offset=self.offset, multimodels=False)      
            self.SaveData(save_dir=self.savePath)
        return self

    def SortDataset(self):
        if self.segmentFeature:
            if self.dateFeature: self.df = self.df.sort(by=[self.segmentFeature, self.dateFeature])
            else: self.df = self.df.sort(by=[self.segmentFeature])

    def StripDataset(self, df):
        if self.yearStart  : df = df.filter(pl.col(self.dateFeature).dt.year()   >= self.yearStart)
        if self.yearEnd    : df = df.filter(pl.col(self.dateFeature).dt.year()   <= self.yearEnd)
        if self.monthStart : df = df.filter(pl.col(self.dateFeature).dt.month()  >= self.monthStart)
        if self.monthEnd   : df = df.filter(pl.col(self.dateFeature).dt.month()  <= self.monthEnd)
        if self.dayStart   : df = df.filter(pl.col(self.dateFeature).dt.day()    >= self.dayStart)
        if self.dayEnd     : df = df.filter(pl.col(self.dateFeature).dt.day()    <= self.dayEnd)
        if self.hourStart  : df = df.filter(pl.col(self.dateFeature).dt.hour()   >= self.hourStart)
        if self.hourEnd    : df = df.filter(pl.col(self.dateFeature).dt.hour()   <= self.hourEnd)
        if self.minuteStart: df = df.filter(pl.col(self.dateFeature).dt.minute() >= self.minuteStart)
        if self.minuteEnd  : df = df.filter(pl.col(self.dateFeature).dt.minute() <= self.minuteEnd)

        return df

    def MachineLearningFillingMissingData(self, model):
        self.SortDataset()
        self.df = self.df.with_columns(pl.col(self.dateFeature).cast(pl.Datetime))
        if self.segmentFeature:
            with self.ProgressBar() as progress:
                dfs = None
                for ele in progress.track(self.df[self.segmentFeature].unique(), description='  Filling data'):
                    df = self.df.filter(pl.col(self.segmentFeature) == ele).clone()
                    df = self.FillDate(df=df)
                    df = df.with_columns(pl.col(self.segmentFeature).fill_null(pl.lit(ele)))
                    if dfs is None: dfs = df
                    else: dfs = pl.concat([dfs, df])
                self.df = dfs
        else: 
            self.df = self.FillDate(df=self.df)
        self.CyclicalPattern()
        self.df = self.GetUsedColumn(df=self.df)
        null_or_not = (self.df.null_count() > 0).rows(named=True)[0]
        target = [key for key, value in null_or_not.items() if value]
        independence = [key for key, value in null_or_not.items() if not value]
        independence.remove(self.dateFeature)
        for t in target:
            with_null = self.df.filter(pl.col(t).is_null()).drop(self.dateFeature)
            without_null = self.df.filter(pl.col(t).is_not_null()).drop(self.dateFeature)
            for item in model_dict:
                if item['type'] == 'MachineLearning':
                    for flag in [item['model'].__name__, *item['alias']]:
                        if model == flag:
                            model = item['model'](modelConfigs=item['config'], save_dir=None)
                            model.modelConfigs['verbosity'] = 0 
                            model.build()
                            model.fit(X_train=without_null[independence].to_numpy(),
                                      y_train=without_null[target].to_numpy())
                            with_null = with_null.with_columns(pl.lit(model.predict(with_null[independence].to_numpy())).alias(t))
                            self.df = self.df.join(other=with_null, on=independence, how="left", suffix="_right")\
                                             .select([
                                                pl.when(pl.col(f'{t}_right').is_not_null())
                                                .then(pl.col(f'{t}_right'))
                                                .otherwise(pl.col(t))
                                                .alias(t),
                                                *independence,
                                                self.dateFeature
                                             ])
                            break
                    else: continue  # only executed if the inner loop did NOT break
                    break  # only executed if the inner loop DID break

    def PolarsFillingMissingData(self, strategy):
        self.SortDataset()
        if self.segmentFeature:
            with self.ProgressBar() as progress:
                dfs = None
                for ele in progress.track(self.df[self.segmentFeature].unique(), description='  Filling data'):
                    df = self.df.filter(pl.col(self.segmentFeature) == ele).clone()
                    df = self.FillDate(df=df)
                    df = df.with_columns(pl.col(self.segmentFeature).fill_null(pl.lit(ele)))
                    for f in [feature for feature in [*self.trainFeatures, self.targetFeatures] if feature != self.segmentFeature]:
                        df = df.with_columns(pl.col(f).fill_null(strategy=strategy))
                    if dfs is None: dfs = df
                    else: dfs = pl.concat([dfs, df])
                self.df = dfs
        else: 
            self.df = self.FillDate(df=self.df)
            for f in [feature for feature in [*self.trainFeatures, self.targetFeatures] if feature != self.segmentFeature]:
                self.df = df.with_columns(pl.col(f).fill_null(strategy=strategy))
            # self.df = self.TimeEncoder(df=self.df).drop_nulls()

    def GetData(self, shuffle):
        if shuffle:
            np.random.shuffle(self.X_train)
            np.random.shuffle(self.y_train)
            np.random.shuffle(self.X_val)
            np.random.shuffle(self.y_val)
            np.random.shuffle(self.X_test)
            np.random.shuffle(self.y_test)
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test

    def GetDataPaths(self, dataPaths=None, extensions=('.csv')):
        if dataPaths: self.dataPaths = dataPaths
        if not isinstance(self.dataPaths, list): self.dataPaths = [self.dataPaths]
        for i in self.dataPaths: 
            if os.path.isdir(i):
                for root, dirs, files in os.walk(i):
                    for file in files:
                        if file.endswith(extensions): 
                            self.dataFilePaths.append(os.path.join(root, file))
            elif i.endswith(extensions) and os.path.exists(i): 
                self.dataFilePaths.append(i)
        assert len(self.dataFilePaths) > 0, 'No csv file(s)'
        self.dataFilePaths = [os.path.abspath(csv) for csv in list_convert(self.dataFilePaths)]

    def ProgressBar(self):
        return Progress("[bright_cyan][progress.description]{task.description}",
                        BarColumn(),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        TextColumn("•Items"),
                        MofNCompleteColumn(), # "{task.completed}/{task.total}",
                        TextColumn("•Remaining"),
                        TimeRemainingColumn(),
                        TextColumn("•Total"),
                        TimeElapsedColumn())

    def ReadFileAddFetures(self, csvs=None, dirAsFeature=0, newColumnName='dir', hasHeader=True):
        if csvs: self.dataFilePaths = [os.path.abspath(csv) for csv in csvs]  
        if dirAsFeature != 0: self.dirAsFeature = dirAsFeature

        if self.dirAsFeature == 0:
            with self.ProgressBar() as progress:
                df = pl.concat([pl.read_csv(source=csv, separator=self.delimiter, has_header=hasHeader, try_parse_dates=True) for csv in progress.track(self.dataFilePaths, description='  Reading data')])
        else:
            dfs = []
            for csv in track(self.dataFilePaths, description='  Reading data'):
                features = [int(p) if p.isdigit() else p for p in csv.split(os.sep)[-self.dirAsFeature-1:-1]]
                df = pl.read_csv(source=csv, separator=self.delimiter, has_header=hasHeader, try_parse_dates=True)
                for idx, f in enumerate(features): 
                    df = df.with_columns(pl.lit(f).alias(f'{newColumnName}{idx}'))
                self.dirFeatures.append(f'{newColumnName}{idx}')
                dfs.append(df)
            df = pl.concat(dfs)
            self.dirFeatures = list(set(self.dirFeatures))
            self.trainFeatures.extend(self.dirFeatures)
    
        if self.df is None: self.df = df
        else: self.df = pl.concat([self.df, df])

        # if self.dateFeature: self.df = self.df.with_columns(pl.col(self.dateFeature).cast(pl.Datetime))

    def TimeIDToDateTime(self, timeIDColumn=None, granularity=1, startTimeId=0):
        if timeIDColumn: self.timeID = timeIDColumn
        if granularity != 1: self.granularity = granularity
        if startTimeId != 0: self.startTimeId = startTimeId

        if not self.timeID: return 

        max_time_id = self.df[self.timeID].max() * self.granularity + self.startTimeId - 24*60
        assert max_time_id <= 0, f'time id max should be {(24*60 - self.startTimeId) / self.granularity} else it will exceed to the next day'
        self.df = self.df.with_columns(pl.col(self.dateFeature).cast(pl.Datetime) + pl.duration(minutes=(pl.col(self.timeID)-1)*self.granularity+self.startTimeId))
    
    def GetUsedColumn(self, df, exclude_date=False):
        if exclude_date: alist = [self.trainFeatures, self.targetFeatures]
        else: alist = [self.dateFeature, self.trainFeatures, self.targetFeatures] 
        return df[[col for i in alist for col in (i if isinstance(i, list) else [i])]]

    def UpdateDateColumnDataType(self, f=pl.Datetime, t=pl.Datetime, dateFormat='%Y-%M-%d'):
        self.df = self.df.with_columns(pl.col(self.dateFeature).str.strptime(f, fmt=dateFormat).cast(t))

    def GetSegmentFeature(self, dirAsFeature=0, splitDirFeature=0, splitFeature=None):
        if dirAsFeature != 0: self.dirAsFeature = dirAsFeature
        if splitDirFeature != 0: self.splitDirFeature = splitDirFeature
        if splitFeature is not None: self.splitFeature = splitFeature

        assert not all([self.dirAsFeature != 0, self.splitFeature is not None])
        self.segmentFeature = self.dirFeatures[self.splitDirFeature] if self.dirAsFeature != 0 and self.splitDirFeature != -1 else self.splitFeature if self.splitFeature else None
        # TODO: consider if data in segmentFeature are number or not. 

    def TimeEncoder(self, df):
        day = 24 * 60 * 60 # Seconds in day  
        year = (365.2425) * day # Seconds in year

        # df = self.FillDate(df=df)
        unix = df[self.dateFeature].to_frame().with_columns(pl.col(self.dateFeature).cast(pl.Utf8).alias('unix_str'))
        unix_time = [time.mktime(parse(t).timetuple()) for t in unix['unix_str'].to_list()]
        df = df.with_columns(pl.lit(unix_time).alias('unix_time'))

        if len(set(df[self.dateFeature].dt.day().to_list())) > 1:
            df = df.with_columns(np.cos((pl.col('unix_time')) * (2 * np.pi / day)).alias('day_cos'))
            df = df.with_columns(np.sin((pl.col('unix_time')) * (2 * np.pi / day)).alias('day_sin'))
            self.trainFeatures.extend(['day_cos', 'day_sin'])
        if len(set(df[self.dateFeature].dt.month().to_list())) > 1:
            df = df.with_columns(np.cos((pl.col('unix_time')) * (2 * np.pi / year)).alias('month_cos'))
            df = df.with_columns(np.sin((pl.col('unix_time')) * (2 * np.pi / year)).alias('month_sin'))
            self.trainFeatures.extend(['month_cos', 'month_sin'])
        
        return df

    def CyclicalPattern(self):
        # assert self.dateFeature is not None
        # if self.segmentFeature:
        #     if self.dateFeature: self.df = self.df.sort(by=[self.segmentFeature, self.dateFeature])
        #     else: self.df = self.df.sort(by=[self.segmentFeature])
        
        # if self.segmentFeature:
        #     dfs = None
        #     for ele in self.df[self.segmentFeature].unique():
        #         df = self.df.filter(pl.col(self.segmentFeature) == ele).clone()
        #         df = self.TimeEncoder(df=df)
        #         if dfs is None: dfs = df
        #         else: dfs = pl.concat([dfs, df])
        #     self.df = dfs.drop_nulls()
        # else: 
        #     self.df = self.TimeEncoder(df=self.df).drop_nulls()
        self.df = self.TimeEncoder(df=self.df)

    def FillDate(self, df=None, low=None, high=None, granularity=None): 
        if not self.dateFeature: return
        if not low: low=self.df[self.dateFeature].min()
        if not high: high=self.df[self.dateFeature].max()
        if not granularity: granularity=self.granularity

        d = pl.date_range(low=low,
                          high=high,
                          interval=timedelta(minutes=granularity),
                          closed='both',
                          name=self.dateFeature).to_frame()
        df = df.join(other=d, 
                     on=self.dateFeature, 
                     how='outer')
        df = self.StripDataset(df=df)
        return df

    def TimeBasedCrossValidation(self, args):
        d, lag, ahead, offset, splitRatio, progressBar = args
        features = []
        labels = []
        if not progressBar:
            for idx in range(len(d)-offset-lag+1):
                feature = d[idx:idx+lag]
                label = d[self.targetFeatures][idx+lag+offset-ahead:idx+lag+offset].to_frame()
                if all(flatten_list(feature.with_columns(pl.all().is_not_null()).rows())) and all(flatten_list(label.with_columns(pl.all().is_not_null()).rows())): 
                    labels.append(np.squeeze(label.to_numpy()))
                    features.append(feature.to_numpy()) 
        else:
            with self.ProgressBar() as progress:
                for idx in progress.track(range(len(d)-offset-lag+1), description='Splitting data'):
                    feature = d[idx:idx+lag]
                    label = d[self.targetFeatures][idx+lag+offset-ahead:idx+lag+offset].to_frame()
                    if all(flatten_list(feature.with_columns(pl.all().is_not_null()).rows())) and all(flatten_list(label.with_columns(pl.all().is_not_null()).rows())): 
                        labels.append(np.squeeze(label.to_numpy()))
                        features.append(feature.to_numpy()) 

        length = len(features)
        if splitRatio[1]==0 and splitRatio[2]==0: 
            train_end = length 
            val_end = length
        elif splitRatio[1]!=0 and splitRatio[2]==0:
            train_end = int(length*splitRatio[0])
            val_end = length
        else:
            train_end = int(length*splitRatio[0])
            val_end = int(length*(splitRatio[0] + splitRatio[1]))
        return [features[0:train_end], features[train_end:val_end], features[val_end:length]], [labels[0:train_end], labels[train_end:val_end], labels[val_end:length]]

    def SplittingData(self, splitRatio, lag, ahead, offset, multimodels=False):
        if self.segmentFeature:
            if self.dateFeature: self.df = self.df.sort(by=[self.segmentFeature, self.dateFeature])
            else: self.df = self.df.sort(by=[self.segmentFeature])
        
        if offset<ahead: offset=ahead

        if self.segmentFeature:
            data = []
            u = self.df[self.segmentFeature].unique()
            with self.ProgressBar() as progress:
                for ele in progress.track(u, description='Splitting jobs'):
                    d = self.df.filter(pl.col(self.segmentFeature) == ele).clone()
                    d = self.FillDate(df=d)
                    d.drop_in_place(self.dateFeature) 
                    data.append([d, lag, ahead, offset, splitRatio, False])
            
            with self.ProgressBar() as progress:
                task_id = progress.add_task("Splitting data", total=len(data))
                with ThreadPool(self.workers) as p:
                    for idx, result in enumerate(p.imap(self.TimeBasedCrossValidation, data)):
                        x = result[0]
                        y = result[1]
                        if multimodels:
                            self.X_train.append(x[0])
                            self.y_train.append(y[0])
                            self.X_val.append(x[1])
                            self.y_val.append(y[1])
                            self.X_test.append(x[2])
                            self.y_test.append(y[2])
                        else:
                            self.X_train.extend(x[0])
                            self.y_train.extend(y[0])
                            self.X_val.extend(x[1])
                            self.y_val.extend(y[1])
                            self.X_test.extend(x[2])
                            self.y_test.extend(y[2])
                        
                        self.num_samples.append({'id' : u[idx],
                                                'train': len(y[0]),
                                                'val': len(y[1]),
                                                'test': len(y[2])})
                        progress.advance(task_id)
        else:
            d = self.df.clone()
            d = self.FillDate(df=d)
            d.drop_in_place(self.dateFeature) 
            
            x, y = self.TimeBasedCrossValidation(args=[d, lag, ahead, offset, splitRatio, True]) 
            self.X_train.extend(x[0])
            self.y_train.extend(y[0])
            self.X_val.extend(x[1])
            self.y_val.extend(y[1])
            self.X_test.extend(x[2])
            self.y_test.extend(y[2])
            self.num_samples.append({'train': len(y[0]),
                                     'val': len(y[1]),
                                     'test': len(y[2])})

        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_val = np.array(self.X_val)
        self.y_val = np.array(self.y_val)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
    
    def SaveData(self, save_dir=None):
        if not save_dir: save_dir = self.save_dir
        save_dir = os.path.join(save_dir, 'values')
        os.makedirs(save_dir, exist_ok=True)

        if self.df is not None: self.df.write_csv(os.path.join(save_dir, 'data_processed.csv'))
        if len(self.num_samples) != 0: 
            with open(os.path.join(save_dir, "num_samples.json"), "w") as final: 
                json.dump(self.num_samples, final, indent=4) 
        np.save(open(os.path.join(save_dir, 'X_train.npy'), 'wb'), self.X_train)
        np.save(open(os.path.join(save_dir, 'y_train.npy'), 'wb'), self.y_train)
        np.save(open(os.path.join(save_dir, 'X_val.npy'), 'wb'), self.X_val)
        np.save(open(os.path.join(save_dir, 'y_val.npy'), 'wb'), self.y_val)
        np.save(open(os.path.join(save_dir, 'X_test.npy'), 'wb'), self.X_test)
        np.save(open(os.path.join(save_dir, 'y_test.npy'), 'wb'), self.y_test)

    def display(self): pass
