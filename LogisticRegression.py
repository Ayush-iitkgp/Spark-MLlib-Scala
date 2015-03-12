from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from numpy import array
import parse
# Load and parse the data

#def parsePoint(line):   # Creating vector(array) with first input as y and others as xi's   
#    values = [float(x) for x in line.split(',')]
#    return LabeledPoint(values[10], values[0:9])


sc = SparkContext("local[4]", "Logistic Regression")      #Initialized SparkContext
data = sc.textFile("/home/ayush/Data /Data for Machine Learning/UCI Adult Data Set/UCI adult.data")  #Created an RDD
parsedData = data.map(parse.parsePoint) #RDD Transformation on the input RDD which is string and converting them to labeled points and each labeled points is a tuple of float(label) and ndrarray(features)

# Build the model
model = LogisticRegressionWithSGD.train(parsedData)   #Pass an RDD to "train" method of class LogisticRegressionwithSGD
#Use model to create output
#model.predict().collect()    # in "predict" method we have to pass an array
#Read Test data

Testdata = sc.textFile("/home/ayush/Data /Data for Machine Learning/UCI Adult Data Set/UCI adult.test")
parsedTestData = Testdata.map(parse.parsePoint)
#predict result for each Test Data

# Evaluating the model on training data

labelsAndPreds = parsedTestData.map(lambda p: (p.label, model.predict(p.features)))  #Taking each array of the RDD of parsedTestData which is a tuple(LabeledPoint) and then calculating its label and features , p is an input to lambda function and p is a tuple point(a LabeledPoint) 
millis2 = int(round(time.time() * 1000))

print labelsAndPreds.collect()
#Print testing Error
testErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedTestData.count())
print("Testing Error = " + str(testErr))

b=millis2-millis1
print "run time(ms) is %f " %b  
