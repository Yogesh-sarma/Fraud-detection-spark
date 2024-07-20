from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql.functions import mean, stddev, col, monotonically_increasing_id
from collections import Counter
# Initialize Spark session
spark = SparkSession.builder.appName("FraudDetection").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
# Define the schema with appropriate data types
schema = StructType([
            StructField("Time", StringType()),
            StructField("V1", FloatType()),
            StructField("V2", FloatType()),
            StructField("V3", FloatType()),
            StructField("V4", FloatType()),
            StructField("V5", FloatType()),
            StructField("V6", FloatType()),
            StructField("V7", FloatType()),
            StructField("V8", FloatType()),
            StructField("V9", FloatType()),
            StructField("V10", FloatType()),
            StructField("V11", FloatType()),
            StructField("V12", FloatType()),
            StructField("V13", FloatType()),
            StructField("V14", FloatType()),
            StructField("V15", FloatType()),
            StructField("V16", FloatType()),
            StructField("V17", FloatType()),
            StructField("V18", FloatType()),
            StructField("V19", FloatType()),
            StructField("V20", FloatType()),
            StructField("V21", FloatType()),
            StructField("V22", FloatType()),
            StructField("V23", FloatType()),
            StructField("V24", FloatType()),
            StructField("V25", FloatType()),
            StructField("V26", FloatType()),
            StructField("V27", FloatType()),
            StructField("V28", FloatType()),
            StructField("Amount", FloatType()),
            StructField("Class", StringType())
        ])
# Define the streaming source (CSV files from local folder)
ssc = StreamingContext(spark.sparkContext, 1)
dstream = ssc.textFileStream("./stream/")
dstream.foreachRDD(process)
ssc.start()
ssc.awaitTermination()

def process(time, rdd):
    data = rdd.filter(lambda line: line != header).map(lambda line: line.split(',')) \
                        .map(lambda tokens: (tokens[0], *[float(token) if token else None for token in tokens[1:29]], tokens[29], float(tokens[30]))) \
                        .toDF(["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Class", "Amount"])
    # Handling missing values
    for col_name in data.columns:
        data = data.filter(col(col_name).isNotNull())
        data.show()
    #There's no missing data
    # Feature list
    feature_list = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"]
    data = data.withColumn("index", monotonically_increasing_id())
    # Detecting outliers
    Outliers_z_score = z_score_method(data, 1, feature_list)
    # Dropping outliers
    df_out3 = data.filter(~col("index").isin(Outliers_z_score))
    print(Outliers_z_score)
    df_out3.show()
    # Register DataFrame as a temporary SQL view
    df_out3.createOrReplaceTempView("transactions")
    # Compute descriptive statistics using Spark SQL
    descriptive_stats = spark.sql("""
                                SELECT
                                Class,
                                COUNT(*) as count,
                                AVG(Amount) as avg_amount,
                                STDDEV(Amount) as std_amount,
                                MIN(Amount) as min_amount,
                                MAX(Amount) as max_amount
                                FROM transactions
                                GROUP BY Class
                            """)
    # Show descriptive statistics
    descriptive_stats.show()
    import matplotlib.pyplot as plt
    # Convert DataFrame to Pandas for visualization
    amount_data = df_out3.select("Class", "Amount").toPandas()
    # Plot histogram
    plt.figure(figsize=(10, 6))
    amount_data[amount_data["Class"] == 0]["Amount"].plot(kind="hist", bins=50,
    alpha=0.5, label="Non-Fraudulent")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.title("Transaction Amount Distribution for non-fraudulent transactions")
    plt.legend()
    plt.show()
    # Convert DataFrame to Pandas for visualization
    amount_data = df_out3.select("Class", "Amount").toPandas()
    # Plot histogram
    plt.figure(figsize=(10, 6))
    amount_data[amount_data["Class"] == 1]["Amount"].plot(kind="hist", bins=50,
    alpha=0.5, label="Fraudulent", color="red")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.title("Transaction Amount Distribution for fraudulent transactions")
    plt.legend()
    plt.show()
    # Convert DataFrame to Pandas for visualization
    time_data = df_out3.select("Class", "Time").toPandas()
    # Plot histogram for transaction time
    plt.figure(figsize=(10, 6))
    time_data[time_data["Class"] == 0]["Time"].plot(kind="hist", bins=50, alpha=0.5,
    label="Non-Fraudulent")
    # time_data[time_data["Class"] == 1]["Time"].plot(kind="hist", bins=50, alpha=0.5, label="Fraudulent")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency")
    plt.title("Transaction Time Distribution for Non-fraudulent transactions")
    plt.legend()
    plt.show()
    # Convert DataFrame to Pandas for visualization
    time_data = df_out3.select("Class", "Time").toPandas()
    # Plot histogram for transaction time
    plt.figure(figsize=(10, 6))
    time_data[time_data["Class"] == 1]["Time"].plot(kind="hist", bins=50, alpha=0.5,
    label="Fraudulent", color="red")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency")
    plt.title("Transaction Time Distribution for fraudulent transactions")
    plt.legend()
    plt.show()
    # Convert DataFrame to Pandas for visualization
    amount_time_data = df_out3.select("Class", "Amount", "Time").toPandas()
    # Plot scatter plot for Amount vs Time
    plt.figure(figsize=(10, 6))
    plt.scatter(amount_time_data[amount_time_data["Class"] == 0]["Time"], amount_time_data[amount_time_data["Class"] == 0]["Amount"], alpha=0.5, label="Non-Fraudulent", c='blue') 
    plt.scatter(amount_time_data[amount_time_data["Class"] == 1]["Time"], amount_time_data[amount_time_data["Class"] == 1]["Amount"], alpha=0.5, label="Fraudulent", c='red')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amount")
    plt.title("Amount vs Time")
    plt.legend()
    plt.show()
    data = df_out3.drop("index")
    data.count()
    # Feature engineering
    # Assemble the feature columns
    assembler = VectorAssembler(inputCols=["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19","V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28","Amount"], outputCol="features")
    data = assembler.transform(data).select("features", "Class")
    data.show()
    # Scale the features
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    scaler_model = scaler.fit(data)
    data = scaler_model.transform(data)
    data.show()
    # Drop original 'Amount' column
    data = data.drop("features")
    # Check the transformed data
    data.show()

    from pyspark.ml.clustering import KMeans
    from pyspark.ml.evaluation import ClusteringEvaluator
    # Train Isolation Forest model
    if_model = KMeans(featuresCol="scaled_features").setK(2).setSeed(1).fit(data)
    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator(featuresCol="scaled_features")
    # Evaluate model
    silhouette = evaluator.evaluate(if_model.transform(data))
    print("Silhouette with squared euclidean distance = " + str(silhouette))
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    # Split data into train and test sets
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    # Train Logistic Regression model
    lr = LogisticRegression(featuresCol='scaled_features', labelCol='Class')
    lr_model = lr.fit(train_data)
    # Make predictions on test data
    lr_predictions = lr_model.transform(test_data)
    # Evaluate model
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",
    labelCol="Class")
    lr_auc = evaluator.evaluate(lr_predictions)
    print(f"Logistic Regression AUC: {lr_auc}")
    lr_predictions.show()
    from pyspark.ml.classification import RandomForestClassifier
    # Train Random Forest model
    rf = RandomForestClassifier(featuresCol='scaled_features', labelCol='Class')
    rf_model = rf.fit(train_data)
    # Make predictions on test data
    rf_predictions = rf_model.transform(test_data)
    # Evaluate model
    rf_auc = evaluator.evaluate(rf_predictions)
    print(f"Random Forest AUC: {rf_auc}")
    from pyspark.ml.classification import GBTClassifier
    # Train Gradient Boosting model
    gbt = GBTClassifier(featuresCol='scaled_features', labelCol='Class', maxIter=10)
    gbt_model = gbt.fit(train_data)
    # Make predictions on test data
    gbt_predictions = gbt_model.transform(test_data)
    # Evaluate model
    gbt_auc = evaluator.evaluate(gbt_predictions)
    print(f"Gradient Boosting AUC: {gbt_auc}")
    gbt_predictions.where("prediction=1.0").count()

    # Compute precision, recall, and F-measure
    gbt_tp = gbt_predictions.where((gbt_predictions.prediction == 0.0) & (gbt_predictions.Class == 0)).count()
    gbt_fp = gbt_predictions.where((gbt_predictions.prediction == 0.0) & (gbt_predictions.Class == 1)).count()
    gbt_tn = gbt_predictions.where((gbt_predictions.prediction == 1.0) & (gbt_predictions.Class == 1)).count()
    gbt_fn = gbt_predictions.where((gbt_predictions.prediction == 1.0) & (gbt_predictions.Class == 0)).count()
    gbt_precision = gbt_tp / (gbt_tp + gbt_fp) if (gbt_tp + gbt_fp) != 0 else 0
    gbt_recall = gbt_tp / (gbt_tp + gbt_fn) if (gbt_tp + gbt_fn) != 0 else 0
    gbt_f_measure = 2 * gbt_precision * gbt_recall / (gbt_precision + gbt_recall) if (gbt_precision +
    gbt_recall) != 0 else 0
    print(f"GBT Regression Precision: {gbt_precision}")
    print(f"GBT Regression Recall: {gbt_recall}")
    print(f"GBT Regression F-measure: {gbt_f_measure}")
    rf_tp = rf_predictions.where((rf_predictions.prediction == 0.0) & (rf_predictions.Class ==0)).count()
    rf_fp = rf_predictions.where((rf_predictions.prediction == 0.0) & (rf_predictions.Class ==1)).count()
    rf_tn = rf_predictions.where((rf_predictions.prediction == 1.0) & (rf_predictions.Class ==1)).count()
    rf_fn = rf_predictions.where((rf_predictions.prediction == 1.0) & (rf_predictions.Class ==0)).count()
    rf_precision = rf_tp / (rf_tp + rf_fp) if (rf_tp + rf_fp) != 0 else 0
    rf_recall = rf_tp / (rf_tp + rf_fn) if (rf_tp + rf_fn) != 0 else 0
    rf_f_measure = 2 * rf_precision * rf_recall / (rf_precision + rf_recall) if (rf_precision + rf_recall)!= 0 else 0
    print(f"GBT Regression Precision: {rf_precision}")
    print(f"GBT Regression Recall: {rf_recall}")
    print(f"GBT Regression F-measure: {rf_f_measure}")
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    # Create ParamGrid for Cross Validation
    paramGrid = (ParamGridBuilder()
    .addGrid(lr.regParam, [0.01, 0.5, 2.0])
    .addGrid(rf.maxDepth, [2, 4, 6])
    .addGrid(gbt.maxDepth, [2, 4, 6])
    .build())
    # Create 5-fold CrossValidator
    cv = CrossValidator(estimator=lr,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=5)
    # Run cross validations
    cvModel = cv.fit(train_data)
    # Use test set to measure the accuracy of our model on new data
    cv_predictions = cvModel.transform(test_data)
    cv_auc = evaluator.evaluate(cv_predictions)
    print(f"Cross-validated Logistic Regression AUC: {cv_auc}")

def z_score_method(data, n, features):
    """
    Takes a PySpark DataFrame data of features and returns a list of indices corresponding to
    the observations containing more than n outliers according to the z-score method.
    """
    outlier_list = []
    threshold = 3
    for column in features:
        # Calculate the mean and standard deviation using PySpark functions
        data_mean = data.agg(mean(col(column))).collect()[0][0]
        data_std = data.agg(stddev(col(column))).collect()[0][0]
        # Calculate the Z-score for the feature column
        data = data.withColumn(column + "_zscore", abs((col(column) - data_mean) / data_std))
        # Determining a list of indices of outliers for feature column
        outlier_list_column = [row[0] for row in data.filter(col(column + "_zscore") >
        threshold).select("index").collect()]
        # Appending the found outlier indices for column to the list of outlier indices
        outlier_list.extend(outlier_list_column)
        # Count the occurrences of each index in the outlier list
        outlier_count = Counter(outlier_list)
        # Select observations containing more than n outliers
        multiple_outliers = [k for k, v in outlier_count.items() if v > n]
    # Calculate the number of outlier records
    outlier_df = data.filter(col(column + "_zscore") > threshold)
    print(f'Total number of outliers is: {outlier_df.count()}')
    return multiple_outliers