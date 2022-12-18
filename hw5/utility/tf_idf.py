import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, HashingTF, IDF
import pyspark.sql.functions as f
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


def tf_idf(target_name, df_clean, num_feature_list):

    hashing = HashingTF(inputCol='filtered', outputCol='raw_features', numFeatures=10)
    idf = IDF(inputCol='raw_features', outputCol='features')
    label_stringIdx = StringIndexer(inputCol = target_name, outputCol = "label")
    lr = LogisticRegression(maxIter=10)

    pipeline = Pipeline(stages= [hashing, idf, label_stringIdx, lr])
    param_grid = ParamGridBuilder().addGrid(hashing.numFeatures, 
                                            num_feature_list).build()
    
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, 
                        evaluator=BinaryClassificationEvaluator(), seed=42)
    model = cv.fit(df_clean)
    params = [{p.name: v for p, v in m.items()} for m in model.getEstimatorParamMaps()]
    
    result = pd.DataFrame.from_dict([{model.getEvaluator().getMetricName(): metric, **ps} 
        for ps, metric in zip(params, model.avgMetrics)])
    
    return result