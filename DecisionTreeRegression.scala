import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Variance
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
object Decision_Tree_Regression {
  def main(args: Array[String]) {

val conf = new SparkConf().setMaster("local[4]").setAppName("Linear Regression with SGD")
val sc = new SparkContext(conf)

// Load and parse the data file
val data = sc.textFile("mllib/data/sample_tree_data.csv")
val parsedData = data.map { line =>
  val parts = line.split(',').map(_.toDouble)
  LabeledPoint(parts(0), Vectors.dense(parts.tail))
}

// Run training algorithm to build the model
val maxDepth = 5
val model = DecisionTree.train(parsedData, Regression, Variance, maxDepth)

// Evaluate model on training examples and compute training error
val valuesAndPreds = parsedData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2)}.mean()
println("training Mean Squared Error = " + MSE)


}
}
