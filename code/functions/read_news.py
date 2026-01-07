import polars as pl

def read_train_news(path='/home/ming/GraduateProject/Data/MINDlarge_train/news.parquet'):
    return pl.read_parquet(path)
def read_dev_news(path='/home/ming/GraduateProject/Data/MINDlarge_train/news.parquet'):
    return pl.read_parquet(path)
def read_news(path='/home/ming/GraduateProject/Data/news.parquet'):
    return pl.read_parquet(path)