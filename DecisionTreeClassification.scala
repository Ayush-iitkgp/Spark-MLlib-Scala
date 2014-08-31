import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Gini

// File name should be same as main Class

object Decision_Tree {
  def main(args: Array[String]){

// Load and parse the data file
val conf = new SparkConf().setMaster("local[4]").setAppName("Decison Tree")
val sc = new SparkContext(conf)
val data = sc.textFile("mllib/data/test.txt")
val parsedData = data.map { line =>
val parts = line.split(' ').map(_.toDouble)
LabeledPoint(parts(0), Vectors.dense(parts.tail))
}
println("No of inputpoints is =  " + parsedData.count())

// Run training algorithm to build the model
val maxDepth = 4
val model = DecisionTree.train(parsedData, Classification, Gini, maxDepth)

// Evaluate model on training examples and compute training error

def tree_predict ( point:LabeledPoint) : Array[Double] = {
  val regression_prediction = model.predict(point.features)
//  println("Regression prediction is = " + regression_prediction)
  var prediction = 0.0
  if (regression_prediction >= 0.5){
         prediction= 1.0
}
   else{
        prediction=0.0
}
//   println("prediction is =  " + prediction)
//   println("label is =  " + point.label)
   println(s"$regression_prediction  $prediction   $point")
   val myList = Array(point.label, prediction)
   return myList
}



val labelAndPreds = parsedData.map(tree_predict)
//println("No of output point is = " +labelAndPreds.count())
//  val regression_prediction = model.predict(point.features)
//  println("Regression prediction is = " + regression_prediction)
//  var prediction =0.0;
//  if (regression_prediction >= 0.5){
// 	 prediction= 1.0                 
//}
//   else{
//	prediction=0.0
//}   
//   println("prediction is =  " + prediction)
//   println("label is =  " + point.label)	
//   (point.label, prediction)
//}
val trainErr = labelAndPreds.filter(r => r(0) != r(1)).count.toDouble / parsedData.count
println("Training Error = " + trainErr)


}

}
