
/**
   * Created by user on 12/4/2020.
   */

import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.clustering.{BisectingKMeans, KMeans}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.expressions.Window

//import org.apache.spark.mllib.clustering.KMeans

object Outliers {


  def main(args: Array[String]): Unit = {

    println(" ")
    println("Hello! I am the outlier detector! :)")
    println(" ")

    val startTime = System.currentTimeMillis()

    //check if user input has one (and only one) argument
    if (args.length < 1 || args.contains(" ")){
      println("No/Wrong argument!..")
      System.exit(-1)
    }
    else{
      println("Argument passed succesfully!")
    }

    //create spark session
    val ss = SparkSession.builder().master("local").appName("OutlierDetection").getOrCreate()

    //Hide spark messages
    import org.apache.log4j.{Level, Logger}
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    println("Reading and processing data...")
    /////read input file
    /////given manually
    //val inputData = ss.read.format("csv").option("header", "false").load("projectB.csv")
    /////case 1 for input args
    val inputData = ss.read.format("csv").option("header", "false").load(args(0) + ".csv")
    /////case 2 for input args (one of them should work --- check at the end!!!!)
    //val currentDir = System.getProperty("user.dir") // get the current directory
    //val inputData = "file://" + currentDir + "/" + args(0)
    var splitData = ss.emptyDataFrame
    //in case file consists of 1 or 2 columns
    //case of 2 columns, since data extraction (_c0 and _c1)
    if( inputData.columns.contains("_c1") ){
      //rename columns o already split data
      splitData = inputData.withColumnRenamed("_c0", "col0").withColumnRenamed("_c1", "col1")
      inputData.unpersist()
    }
    //case of 1 column
    else {
      //split data into columns, since data extraction (_c0)
        splitData = inputData.withColumn("temp", split(col("_c0"), "\\,")).select(
        (0 until 2).map(i => col("temp").getItem(i).as(s"col$i")): _*
      )
      inputData.unpersist()
    }
    //cast columns to double
    val doubleData = splitData.withColumn("col0", col("col0").cast("Double")).withColumn("col1", col("col1").cast("Double"))
    //drop columns that do not contain both fields
    val cleanData = doubleData.na.drop() //doubleData
    doubleData.unpersist()
    //val cleanedData = cleanData.select("col0", "col1")
    //cleanData.unpersist()
    //Normalize data
    val vectorizeCol = udf((v: Double) => Vectors.dense(Array(v)))
    val VectorizedDF = cleanData // cleanedData
      .withColumn("col0Vec", vectorizeCol(cleanData("col0"))) //cleanedData
      .withColumn("col1Vec", vectorizeCol(cleanData("col1"))) //cleanedData
    //using MinMaxScalerfor col0
    val scalerCol0 = new MinMaxScaler()
      .setInputCol("col0Vec")
      .setOutputCol("col0_norm")
      .setMax(1)
      .setMin(0)
    //using MinMaxScalerfor col1
    val scalerCol1 = new MinMaxScaler()
      .setInputCol("col1Vec")
      .setOutputCol("col1_norm")
      .setMax(1)
      .setMin(0)
    val NormTempDF = scalerCol0
      .fit(VectorizedDF)
      .transform(VectorizedDF)
    val NormalizedDF = scalerCol1
      .fit(NormTempDF)
      .transform(NormTempDF)

    val cols = Array("col0_norm", "col1_norm")
    val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
    val featureDf = assembler.transform(NormalizedDF)

    var k = 0 //k -> number of clusters of k-means

    //count rows of dataframe
    val rowsCount = featureDf.count()
    //based on row count we determine the number of cluster that the k-means will run
    //if rows < 1000 we set k = 10
    if ( rowsCount < 10000 ){
      k = 100
    }
    //if rows < 50000 we set k = 500
    else if ( rowsCount < 50000 ) {
      k = 500
    }
    //if rows >=50000 we divide with 1000 and add 500 in order to avoid running k-means with a great number of clusters
    else{
      val k = 500 + (rowsCount / 1000).toInt
    }

    /* =============================================================================== */
    /*                                   K - Means                                     */
    /* =============================================================================== */
    println("Running K-Means with " + k + " clusters")

    val kmeans = new KMeans()
      .setK(k)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setMaxIter(1000000000) // :D

    val kmeansModel = kmeans.fit(featureDf)
    //kmeansModel.clusterCenters.foreach(println)
    var predictDf = kmeansModel.transform(featureDf)

    //Count number of points of each cluster and check for outlier clusters////////////////////////////////////////////////////////////////////////////////////
    //creates dataframe with cols: cluster id, number of points of cluster
    var clusterPointQtyDf = predictDf.groupBy("prediction")
      .count
    //rename columns
    val newColumns = Seq("cluster_id","pointsQTY")
    clusterPointQtyDf = clusterPointQtyDf.toDF(newColumns:_*)
    //compute division |Cb|/|Cb+1| for each cluster//////////////////////////////////////
    val w2 = Window.orderBy(col("pointsQTY").desc)
    clusterPointQtyDf = clusterPointQtyDf.withColumn("lead", lead(clusterPointQtyDf("pointsQTY"), 1) over (w2))
    //last row has value=null
    clusterPointQtyDf = clusterPointQtyDf.withColumn("div", clusterPointQtyDf("pointsQTY") / clusterPointQtyDf("lead")) //.drop("lead")
    //move each div result one row above in order to match to cluster id
    val w3 = Window.rowsBetween(
      Window.currentRow, // To current row
      Window.unboundedFollowing // Take all rows from the beginning of frame
    )

    clusterPointQtyDf = clusterPointQtyDf.withColumn("div", clusterPointQtyDf("pointsQTY") / clusterPointQtyDf("lead"))

    //determine thresholds and check for outlier clusters//////////////////
    clusterPointQtyDf = clusterPointQtyDf.withColumn("id",monotonicallyIncreasingId)

    //parameter >= 5 to determine how many times an outlier cluster is iin comparison with the rest. After experiments we conducted that a cluster 3 times smaller can be considered as an outlier cluster

    import ss.implicits._
    val thresholdDivDf = clusterPointQtyDf.filter($"div" >= 5).select($"id").toDF()

    if (thresholdDivDf.rdd.isEmpty == false ) {

      val thresholdDiv = thresholdDivDf.first.getLong(0) //division threshold

      clusterPointQtyDf = clusterPointQtyDf
        .withColumn("IsOutlierFirstCheck", when(col("id") > thresholdDiv, true).otherwise(false))
      //clusterPointQtyDf.show(k)

      //Determine points that belong in outlier clusters//////////////////
      //join with points dataframe
      var Df = predictDf.join(clusterPointQtyDf,(col("prediction") === col("cluster_id")))
      println(" ")
      println("Outliers from 1st check (Cluster-Based Approach): ")

      //print them as points
      val firstCheckRDD = Df.filter(col("IsOutlierFirstCheck") === true).select("col0","col1", "div").rdd.map(r => (r(0),r(1))).collect()

      firstCheckRDD.foreach(println)
      print("\n")
    }
      //in case there are no outlier clusters
    else{
      println("Outliers from 1st check: None, No Outlier-Clusters formed")
    }

    //2. Determine outlier points using Euclidean distance///////////////////////////////////////////////
    // UDF that calculates for each point distance from each cluster center

    val dstartTime = System.currentTimeMillis()

    val distFromCenter = udf((features: Vector, c: Int) => Vectors.sqdist(features, kmeansModel.clusterCenters(c)))

    //Calculate each point's distance point from it's cluster center
    var distancesDF = predictDf.withColumn("distanceFromCenter", distFromCenter(col("features"),col("prediction")))   //($"features", $"prediction"))

    //Calculate avg distance from center
    var avgDistDf = distancesDF.groupBy("prediction").avg("distanceFromCenter").as("avgDist")

    //rename columns
    val newColumns2 = Seq("cluster_id","avgDist")
    avgDistDf = avgDistDf.toDF(newColumns2:_*)

    distancesDF = distancesDF.join(avgDistDf,
      (col("prediction") === col("cluster_id")))

    //after thorough experiments we deducted that 5.5 times distance of a point from the centroid in each cluster in comparison to the average distance, is a safe distance to be considered as ann outlier
    val parameterB = 8
    distancesDF =  distancesDF.withColumn("div",distancesDF("distanceFromCenter")/distancesDF("avgDist"))
    distancesDF = distancesDF.withColumn("IsOutlierSecondCheck", when(col("div") >= parameterB, true).otherwise(false))

    //print out 2nd check outliers
    println("Outliers from 2nd check (Distance-Based Approach): ")
    //print them as a df
    //val secondCheckDf = distancesDF.filter(col("IsOutlierSecondCheck") === true).select("col0", "col1").show(100, false)
    //print them as points

    val secondCheckRDD = distancesDF.filter(col("IsOutlierSecondCheck") === true).select("col0","col1").rdd.map(r => (r(0),r(1))).collect()
    if ( secondCheckRDD.isEmpty == false ) {
      secondCheckRDD.foreach(println)
    }
    else{
      println("Outliers from 2nd check: None. ")
    }

    //calculate running time
    val endTime = System.currentTimeMillis()
    val exeTime = ( endTime - startTime ) / 1000
    val sec = exeTime % 60
    val min = ( exeTime - sec ) / 60
    println("")
    println("Execution time: " + min + " min. " + sec + " sec. ")

  }
}
