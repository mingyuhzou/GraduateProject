import polars as pl
from .read_behaviors import read_train_behaviors,read_dev_behaviors

def get_users_earliest_hist(data):
    """
    只保留每个 user_id 最旧的一次 impression 的真实点击
    :param data: user_id, time, history, impressions
    :return: user_id, hist(list[str])
    """
    # 1. 每个 user 只保留 time 最小的一条
    data_oldest = (
        data
        .sort("time")
        .group_by("user_id")
        .agg(pl.all().first())
    )

    # 2. 从 impressions 中抽取正样本
    click = (
        data_oldest
        .select(["user_id", "impressions"])
        .explode("impressions")
        .with_columns([
            pl.col("impressions")
            .str.split("-")
            .list.get(0)
            .alias("article_id"),
            pl.col("impressions")
            .str.split("-")
            .list.get(1)
            .cast(pl.Int8)
            .alias("label"),
        ])
        .filter(pl.col("label") == 1)
    )

    # 3. 聚合为 user → hist
    user_hist = (
        click
        .group_by("user_id")
        .agg(pl.col("article_id").alias("hist"))
    )

    return user_hist


def valid_popularity_recall(pred, topk=5):
    gt = get_users_earliest_hist(read_dev_behaviors())
    data = pred.join(gt, on="user_id", how="inner")

    for i in range(1, topk + 1):
        k = i * 10

        recall_k = (
            data
            .with_columns(
                pl.col("rec_list").list.slice(0, k).alias("rec_k")
            )
            .explode("rec_k")
            .with_columns(
                pl.col("rec_k").is_in(pl.col("hist")).alias("hit_flag")
            )
            .group_by("user_id")
            .agg(
                pl.sum("hit_flag").alias("hit"),
                pl.first("hist").list.len().alias("gt")
            )
            .filter(pl.col("gt") > 0)
            .select((pl.col("hit") / pl.col("gt")).mean())
            .item()
        )

        print(f"Recall@{k}", recall_k)

def get_impression_gt(data):
    """
    impression 级 GT
    return: impr_id, user_id, gt(list[str])
    """
    gt = (
        data
        .with_row_count("impr_id")
        .select(["impr_id", "user_id", "impressions"])
        .explode("impressions")
        .with_columns([
            pl.col("impressions").str.split("-").list.get(0).alias("news_id"),
            pl.col("impressions").str.split("-").list.get(1).cast(pl.Int8).alias("label"),
        ])
        .filter(pl.col("label") == 1)
        .group_by(["impr_id", "user_id"])
        .agg(pl.col("news_id").alias("gt"))
    )
    return gt


def valid_recall_impression(pred, topk=5):
    gt = get_impression_gt(read_dev_behaviors())

    data = pred.join(
        gt,
        on=["impr_id", "user_id"],
        how="inner"
    )

    for i in range(1, topk + 1):
        k = i * 10

        recall_k = (
            data
            # 1. 取前 k 个召回结果
            .with_columns(
                pl.col("rec_list").list.slice(0, k).alias("rec_k")
            )
            # 2. 计算命中个数 |rec@k ∩ gt|
            .with_columns(
                pl.col("rec_k")
                .list.eval(pl.element().is_in(pl.col("gt")))
                .list.sum()
                .alias("hit")
            )
            # 3. impression 级 recall
            .with_columns(
                (pl.col("hit") / pl.col("gt").list.len()).alias("recall")
            )
            # 4. 所有 impression 取平均
            .select(pl.col("recall").mean())
            .item()
        )

        print(f"Recall@{k}: {recall_k:.4f}")



