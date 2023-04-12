import os
import numpy as np
import polars as pl
from rich.progress import track
from datetime import timedelta
from multiprocessing.pool import ThreadPool

from utils.general import yaml_load
from utils.general import list_convert
from utils.general import flatten_list

from rich.progress import Progress
from rich.progress import BarColumn 
from rich.progress import MofNCompleteColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn

class DatasetController():
    def __init__(self, configsPath=None, dirAsFeature=0, splitDirFeature=0, splitFeature=None, granularity=1, startTimeId=0, workers=8, splitRatio=(0.7, 0.2, 0.1), lag=3, ahead=1, offset=1, savePath='.'):
        
        """ Read data config """
        self.dataConfigs = yaml_load(configsPath)

        self.dataPaths = self.dataConfigs['data']
        self.dateFeature = self.dataConfigs['date']
        self.timeID = self.dataConfigs['time_id']
        self.targetFeatures = self.dataConfigs['target']
        self.delimiter = self.dataConfigs['delimiter']
        self.trainFeatures = list_convert(self.dataConfigs['features'])
        
        self.configsPath = configsPath
        self.dirAsFeature = dirAsFeature
        self.splitDirFeature = splitDirFeature
        self.splitFeature = splitFeature
        self.workers = workers
        self.granularity = granularity
        self.startTimeId = startTimeId
        self.splitRatio = splitRatio
        self.lag = lag
        self.ahead = ahead
        self.offset = offset
        self.savePath = savePath

        self.df = None
        self.dataFilePaths = []
        self.dirFeatures = []
        self.segmentFeature = None
        self.X_train = []
        self.y_train = []
        self.X_val = []
        self.y_val = []
        self.X_test = []
        self.y_test = []

        self.num_samples = []

    def execute(self):
        assert self.dataPaths is not None
        self.GetDataPaths(self.dataPaths)
        self.ReadFileAddFetures(csvs=self.dataFilePaths, dirAsFeature=self.dirAsFeature, hasHeader=True)
        self.TimeIDToDateTime(timeIDColumn=self.timeID, granularity=self.granularity, startTimeId=self.startTimeId)
        self.GetUsedColumn()
        self.GetSegmentFeature(dirAsFeature=self.dirAsFeature, splitDirFeature=self.splitDirFeature, splitFeature=self.splitFeature)
        self.SplittingData(splitRatio=self.splitRatio, lag=self.lag, ahead=self.ahead, offset=self.offset, multimodels=False)      
        self.SaveData(save_dir=self.savePath)
        return self

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
        # if delimiter != ',': self.delimiter = delimiter 

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

    def TimeIDToDateTime(self, timeIDColumn=None, granularity=1, startTimeId=0):
        if timeIDColumn: self.timeID = timeIDColumn
        if granularity != 1: self.granularity = granularity
        if startTimeId != 0: self.startTimeId = startTimeId

        if not self.timeID: return 

        max_time_id = self.df[self.timeID].max() * self.granularity + self.startTimeId - 24*60
        assert max_time_id <= 0, f'time id max should be {(24*60 - self.startTimeId) / self.granularity} else it will exceed to the next day'
        self.df = self.df.with_columns(pl.col(self.dateFeature).cast(pl.Datetime) + pl.duration(minutes=(pl.col(self.timeID)-1)*self.granularity+self.startTimeId))
    
    def GetUsedColumn(self):
        self.df = self.df[[col for i in [self.dateFeature, self.trainFeatures, self.targetFeatures] for col in (i if isinstance(i, list) else [i])]]

    def UpdateDateColumnDataType(self, dateFormat='%Y-%M-%d'):
        self.df = self.df.with_columns(pl.col(self.dateFeature).str.strptime(pl.Date, fmt=dateFormat).cast(pl.Datetime))

    def GetSegmentFeature(self, dirAsFeature=0, splitDirFeature=0, splitFeature=None):
        if dirAsFeature != 0: self.dirAsFeature = dirAsFeature
        if splitDirFeature != 0: self.splitDirFeature = splitDirFeature
        if splitFeature is not None: self.splitFeature = splitFeature

        assert not all([self.dirAsFeature != 0, self.splitFeature is not None])
        self.segmentFeature = self.dirFeatures[self.splitDirFeature] if self.dirAsFeature != 0 and self.splitDirFeature != -1 else self.splitFeature if self.splitFeature else None
        # TODO: consider if data in segmentFeature are number or not. 

    def CyclicalPattern(self): pass

    def FillDate(self, df=None, low=None, high=None): 
        # TODO: cut date
        if not self.dateFeature: return
        # if df.is_empty(): df=self.df
        if not low: low=self.df[self.dateFeature].min()
        if not high: high=self.df[self.dateFeature].max()

        d = pl.date_range(low=low,
                          high=high,
                          interval=timedelta(minutes=self.granularity),
                          closed='both',
                          name=self.dateFeature).to_frame()
        df = df.join(other=d, 
                     on=self.dateFeature, 
                     how='outer')
        return df

    def TimeBasedCrossValidation(self, args):
        d, lag, ahead, offset, splitRatio = args
        features = []
        labels = []
        for idx in range(len(d)-offset-lag+1):
            feature = d[idx:idx+lag]
            label = d[self.targetFeatures][idx+lag+offset-ahead:idx+lag+offset].to_frame()
            if all(flatten_list(feature.with_columns(pl.all().is_not_null()).rows())) and all(flatten_list(label.with_columns(pl.all().is_not_null()).rows())): 
                # print(feature)
                # print(d[idx+lag+offset-ahead:idx+lag+offset])
                # print('======================================================================================================')
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
            # if self.workers != 1: 
            data = []
            with self.ProgressBar() as progress:
                for ele in progress.track(self.df[self.segmentFeature].unique(), description='Splitting jobs'):
                    d = self.df.filter(pl.col(self.segmentFeature) == ele).clone()
                    d = self.FillDate(df=d)
                    d.drop_in_place(self.dateFeature) 
                    data.append([d, lag, ahead, offset, splitRatio])
            
            with self.ProgressBar() as progress:
                task_id = progress.add_task("Splitting data", total=len(data))
                with ThreadPool(self.workers) as p:
                    for result in p.imap(self.TimeBasedCrossValidation, data):
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
                        
                        self.num_samples.append({'id' : ele,
                                                'train': len(y[0]),
                                                'val': len(y[1]),
                                                'test': len(y[2])})
                        progress.advance(task_id)
        else:
            d = self.df.clone()
            d = self.FillDate(df=d)
            d.drop_in_place(self.dateFeature) 
            
            x, y = self.TimeBasedCrossValidation(args=[d, lag, ahead, offset, splitRatio]) 
            self.X_train.extend(x[0])
            self.y_train.extend(y[0])
            self.X_val.extend(x[1])
            self.y_val.extend(y[1])
            self.X_test.extend(x[2])
            self.y_test.extend(y[2])

        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_val = np.array(self.X_val)
        self.y_val = np.array(self.y_val)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
    
    def SaveData(self, save_dir):
        save_dir = os.path.join(save_dir, 'values')
        os.makedirs(save_dir, exist_ok=True)
        np.save(open(os.path.join(save_dir, 'X_train.npy'), 'wb'), self.X_train)
        np.save(open(os.path.join(save_dir, 'y_train.npy'), 'wb'), self.y_train)
        np.save(open(os.path.join(save_dir, 'X_val.npy'), 'wb'), self.X_val)
        np.save(open(os.path.join(save_dir, 'y_val.npy'), 'wb'), self.y_val)
        np.save(open(os.path.join(save_dir, 'X_test.npy'), 'wb'), self.X_test)
        np.save(open(os.path.join(save_dir, 'y_test.npy'), 'wb'), self.y_test)

    def display(self): pass
