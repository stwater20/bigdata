import pandas as pd
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, HiveContext, functions
from pyspark.sql.functions import mean as _mean, stddev as _stddev, col, udf, concat_ws
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, MinMaxScaler

HW1_PATH = "/home/ubuntu/Desktop/hw0/"

# Spark configuration
conf = SparkConf().setAppName("HW0").setMaster("spark://192.168.225.128:7077")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.appName("HW0").getOrCreate()
sqlContext = HiveContext(sc)

# Function to save results to a file
def save_to_file(content, file_name):
    with open(file_name, "a") as f:
        f.write(content + "\n")

# Function to get min, max, and count of a column and return as a formatted string
def get_col_agg_values(col_name, df):
    max_val = df.agg({col_name: "max"}).collect()[0]["max("+col_name+")"]
    min_val = df.agg({col_name: "min"}).collect()[0]["min("+col_name+")"]
    count = df.agg({col_name: "count"}).collect()[0]["count("+col_name+")"]
    return f"{col_name}: Min: {min_val}, Max: {max_val}, Count: {count}"

# Function to get mean and standard deviation of a column and return as a formatted string
def get_mean_and_std_dev(col_name, df):
    std_dev_and_mean = df.select(_mean(col(col_name)).alias("mean"),
                                 _stddev(col(col_name)).alias("std")).collect()
    mean = std_dev_and_mean[0]["mean"]
    std = std_dev_and_mean[0]["std"]
    return f"{col_name}: Mean: {mean}, Std Dev: {std}"

# Main function
def main():
    df = sqlContext.read \
        .format('com.databricks.spark.csv') \
        .options(header='true', delimiter=';') \
        .load(HW1_PATH + 'household_power_consumption.txt')
    df = df.replace({"?": None})  # replace ? with None
    df = df.na.drop()  # drop the values with None

    df = df.withColumn("Global_active_power", df["Global_active_power"].cast("float"))
    df = df.withColumn("Global_reactive_power", df["Global_reactive_power"].cast("float"))
    df = df.withColumn("Voltage", df["Voltage"].cast("float"))
    df = df.withColumn("Global_intensity", df["Global_intensity"].cast("float"))

    # Q1: Save min, max, and count of the specified columns to a file (4 columns, 3 values each)
    q1_output_file = "Q1_output.txt"
    with open(q1_output_file, "w") as f:
        f.write("Task 1: Min, Max, Count of 4 columns\n")
        f.write(get_col_agg_values("Global_active_power", df) + "\n")
        f.write(get_col_agg_values("Global_reactive_power", df) + "\n")
        f.write(get_col_agg_values("Voltage", df) + "\n")
        f.write(get_col_agg_values("Global_intensity", df) + "\n")

    # Q2: Save mean and standard deviation of the specified columns to a file (4 columns, 2 values each)
    q2_output_file = "Q2_output.txt"
    with open(q2_output_file, "w") as f:
        f.write("Task 2: Mean and Std Dev of 4 columns\n")
        f.write(get_mean_and_std_dev("Global_active_power", df) + "\n")
        f.write(get_mean_and_std_dev("Global_reactive_power", df) + "\n")
        f.write(get_mean_and_std_dev("Voltage", df) + "\n")
        f.write(get_mean_and_std_dev("Global_intensity", df) + "\n")

    # Q3: Perform Min-Max normalization and save the results to a file (one file, one row per line)
    col_list = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity"]

    assemblers = [VectorAssembler(inputCols=[col], outputCol=col + "_vec") for col in col_list]
    scalers = [MinMaxScaler(inputCol=col + "_vec", outputCol="normalized_" + col) for col in col_list]
    pipeline = Pipeline(stages=assemblers + scalers)
    scaler_model = pipeline.fit(df)
    scaled_data = scaler_model.transform(df)
    scaled_data = scaled_data.select(scaled_data.columns[-4:])

    # UDF to get the first element from vector
    get_first_ele = udf(lambda vec: str(vec[0]), StringType())
    normalized_cols = scaled_data.select([get_first_ele(f"normalized_{col}").alias(f"normalized_{col}") for col in col_list])

    # Concatenate the normalized values into a single string per row
    normalized_output_file = "Q3_output.txt"
    normalized_cols.withColumn("normalized_output", concat_ws(",", *[functions.col(f"normalized_{x}") for x in col_list])) \
                    .select("normalized_output") \
                    .write.mode("overwrite").text(normalized_output_file)

if __name__ == "__main__":
    main()
