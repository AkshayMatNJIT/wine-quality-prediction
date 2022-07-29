#import findspark
#findspark.init()
#findspark.find()

#Loading the libraries
import pyspark
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession	
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#Starting the spark session
conf = pyspark.SparkConf().setAppName('winequality').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)

path  = sys.argv[1]
#loading the validation dataset
val = spark.read.format("csv").load(path, header = True , sep=";")
val.printSchema()
val.show()

    
#changing the 'quality' column name to 'label'
for col_name in val.columns[1:-1]+['""""quality"""""']:
    val = val.withColumn(col_name, col(col_name).cast('float'))
val = val.withColumnRenamed('""""quality"""""', "label")

#getting the features and label seperately and converting it to numpy array
features =np.array(val.select(val.columns[1:-1]).collect())
label = np.array(val.select('label').collect())

#creating the feature vector
VectorAssembler = VectorAssembler(inputCols = val.columns[1:-1] , outputCol = 'features')
df_tr = VectorAssembler.transform(val)
df_tr = df_tr.select(['features','label'])

#The following function creates the labeledpoint and parallelize it to convert it into RDD
def to_labeled_point(sc, features, labels, categorical=False):
    labeled_points = []
    for x, y in zip(features, labels):        
        lp = LabeledPoint(y, x)
        labeled_points.append(lp)
    return sc.parallelize(labeled_points) 

#rdd converted dataset
dataset = to_labeled_point(sc, features, label)

#loading the model from s3
RFModel = RandomForestModel.load(sc, "/winepredict/trainingmodel.model/")

print("model loaded successfully")
predictions = RFModel.predict(dataset.map(lambda x: x.features))

#getting a RDD of label and predictions
labelsAndPredictions = dataset.map(lambda lp: lp.label).zip(predictions)
 
labelsAndPredictions_df = labelsAndPredictions.toDF()
#cpnverting rdd ==> spark dataframe ==> pandas dataframe 
labelpred = labelsAndPredictions.toDF(["label", "Prediction"])
labelpred.show()
labelpred_df = labelpred.toPandas()


#Calculating the F1score
F1score = f1_score(labelpred_df['label'], labelpred_df['Prediction'], average='micro')
print("F1- score: ", F1score)
print(confusion_matrix(labelpred_df['label'],labelpred_df['Prediction']))
print(classification_report(labelpred_df['label'],labelpred_df['Prediction']))
print("Accuracy" , accuracy_score(labelpred_df['label'], labelpred_df['Prediction']))

#calculating the test error
testErr = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(dataset.count())    
print('Test Error = ' + str(testErr))

