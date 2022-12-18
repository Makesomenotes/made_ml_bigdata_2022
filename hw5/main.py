import os

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

from utility import clean_text, tf_idf

DATA_PATH = r'./data/toxicity'
TRAIN_PATH = os.path.join(DATA_PATH, 'train.csv')


def main():
    spark = SparkSession.builder \
            .master("local[1]") \
            .appName("HTF") \
            .getOrCreate() \

    sc = spark.sparkContext
    df = spark.read.option("header",True).option("quote", "\"").option("escape", "\"").option("multiline", True).option("delimiter", ',').csv(TRAIN_PATH)
    df_clean = clean_text(df)
    # tf_idf
    for target in ['toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        target_df = tf_idf(target, df_clean, [10, 50, 200])
        path_to_save = f'./data/tf_idf/tf_idf_{target}.csv'
        target_df.to_csv(path_to_save)


if __name__ == '__main__':
    main()
