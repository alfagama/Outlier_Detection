
/**
   * Created by user on 12/4/2020.
   */

import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.clustering.{BisectingKMeans, KMeans}
import org.apache.spark.ml.linalg.{Vector, Vectors}
//import org.apache.spark.mllib.clustering.KMeans


object ProjectB_kmeans_outlier_detection {


   println("Hello! I am the outlier detector! :)")
   println("Please pass as an argument the input file you wish to run!")

     def main(args: Array[String]): Unit = {

       val startTime = System.currentTimeMillis()

       /*
       //check if user input has one (and only one)  
argument/////////////////////////////////////////////////////////////
       if (args.length < 1 || args.contains(" ")){
         println("No/Wrong argument!..")
         System.exit(-1)
       }
       */

       //create spark session  
///////////////////////////////////////////////////////////////////////////////////////////
       val ss =  
SparkSession.builder().master("local").appName("tfidfApp").getOrCreate()
       import ss.implicits._  // For implicit conversions like  
converting RDDs to DataFrames
       val currentDir = System.getProperty("user.dir")  // get the  
current directory

       println("Reading data...")
       println

       //read input file  
////////////////////////////////////////////////////////////////////////////////////////////////
       val inputData = ss.read.format("csv").option("header",  
"false").load("projectB.csv")
       //case 1 for input args
       //val inputData = ss.read.format("csv").option("header",  
"false").load(args(0) + ".csv")
       //case 2 for input args (one of them should work --- check at  
the end!!!!)
       //val inputFile = "file://" + currentDir + "/" + args(0)
       //inputData.printSchema()
       //inputData.show(5, false)

       /*
       if( inputData.columns.contains("_c1") ){
         //split data into columns  
////////////////////////////////////////////////////////////////////////////////////////
         val doubleData = inputData.withColumn("_c0",  
col("_c0").cast("Double")).withColumn("_c1", col("_c1").cast("Double"))
         doubleData.printSchema()
         println("data converted to Double!")

         //drop columns that do not contain both fields  
///////////////////////////////////////////////////////////////////
         val cleanData = doubleData.na.drop() //doubleData
         val cleanedData = cleanData.select("_c0", "_c1")
         //doubleData.unpersist()
         println("data cleaned!")
       }else{}
       */

         //split data into columns  
////////////////////////////////////////////////////////////////////////////////////////
         val splitData = inputData.withColumn("temp",  
split(col("_c0"), "\\,")).select(
           (0 until 2).map(i => col("temp").getItem(i).as(s"col$i")): _*
         )
         inputData.unpersist()
         splitData.show(5, false)

         //convert data from Sting to Double  
//////////////////////////////////////////////////////////////////////////////
         /*
         //not used udf toDouble
         val toDouble = udf[Double, String]( _.toDouble)
         val doubleData2 = splitData
           .withColumn("col0", toDouble(splitData("col0")))
           .withColumn("col1", toDouble(splitData("col1")))
           .select("col0", "col1")
         doubleData2.printSchema()
         */
         //not used Seq toDouble
         /*
         splitData.select(Seq("col0", "col1").map( c =>  
col(c).cast("Double")): _*)
         splitData.printSchema()
         */
         val doubleData = splitData.withColumn("col0",  
col("col0").cast("Double")).withColumn("col1",  
col("col1").cast("Double"))
         //doubleData.printSchema()
         splitData.unpersist()
         doubleData.show(5, false)
         //drop columns that do not contain both fields  
///////////////////////////////////////////////////////////////////
         val cleanData = doubleData.na.drop() //doubleData
         doubleData.unpersist()
         cleanData.show(5, false)
         val cleanedData = cleanData.select("col0", "col1")
         cleanData.unpersist()
         cleanedData.show(5, false)
       //apply normalization to data  
////////////////////////////////////////////////////////////////////////////////////
       def call_assembler(data: DataFrame) = {
         val assembler = new VectorAssembler()
           .setInputCols(data.columns)
           .setOutputCol("features")
         assembler.transform(data)
       }
       val assemblerData = call_assembler(cleanedData).select("features")
       cleanedData.unpersist()
       assemblerData.show(4, false)
       val normalizer = new Normalizer()
         .setInputCol("features")
         .setOutputCol("normFeatures")
         .setP(2.0)
       val normalizedData =  
normalizer.transform(assemblerData).select("features", "normFeatures")
       assemblerData.unpersist()
       normalizedData.show(6, false)
       val kmeansData = normalizedData
           .withColumnRenamed("features", "oldfeatures")
           .withColumnRenamed("normFeatures", "features")
       //kmeansData.printSchema()
       //kmeansData.show(11, false)
       normalizedData.unpersist()

       /*  
===============================================================================  
*/
       /*                                   K - Means                   
                    */
       /*  
===============================================================================  
*/
       kmeansData.show(5, false)
       println("Running K-Means...")

       //train k-means model
       val kmeans = new  
KMeans().setK(5).setFeaturesCol("features").setPredictionCol("cluster_id")  
//"setSeed(1L) //.setMaxIter(63436363).setSeed(2L)
       val modelkm = kmeans.fit(kmeansData)
       //make predictions
       val predictions = modelkm.transform(kmeansData)

       //predictions.show(5, false)
       /*
       //evaluate clustering by computing Silhouette score
       val evaluator = new ClusteringEvaluator()
       val silhouette = evaluator.evaluate(predictions)
       println(s"Silhouette with squared euclidean distance = $silhouette")
       //show results
       println("Cluster Centers: ")
       modelkm.clusterCenters.foreach(println)
       */
       /*  
===============================================================================  
*/
       /*                              Bisecting K - Means              
                    */
       /*  
===============================================================================  
*/
       /*
       println("Running Bisecting K-Means...")

       //train a bisecting k-means model
       val bkm = new BisectingKMeans().setK(5).setSeed(1)
       val modelbkm = bkm.fit(kmeansData)
       //evaluate clustering
       val cost = modelbkm.computeCost(kmeansData)
       println(s"Within Set Sum of Squared Errors = $cost")
       //show the result
       println("Cluster Centers: ")
       val centers = modelbkm.clusterCenters
       centers.foreach(println)
       */



     }

}

