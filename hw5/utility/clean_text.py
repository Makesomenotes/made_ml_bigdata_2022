from pyspark.ml.feature import StopWordsRemover, Tokenizer
import pyspark.sql.functions as f


def clean_text(df):
    # lower case
    df_lower = df.withColumn('comment_text', f.lower(f.col('comment_text')))
    # stop words
    tokenizer = Tokenizer(inputCol='comment_text', outputCol='words')
    df_to_clean = tokenizer.transform(df_lower)
    remover = StopWordsRemover(inputCol='words', outputCol='filtered')
    df_clean = remover.transform(df_to_clean)
    
    return df_clean
