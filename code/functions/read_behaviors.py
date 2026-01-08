import polars as pl

def read_train_behaviors(path='/home/ming/GraduateProject/Data/MINDlarge_train/behaviors.parquet'):
    return pl.read_parquet(path)
def read_dev_behaviors(path='/home/ming/GraduateProject/Data/MINDlarge_dev/behaviors_1.parquet'):
    return pl.read_parquet(path)
def  read_behaviors():
    return pl.concat([read_train_behaviors(),read_dev_behaviors()],how='vertical')