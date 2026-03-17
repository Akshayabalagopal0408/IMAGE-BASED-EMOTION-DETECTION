from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, FloatType, StringType
import numpy as np
import cv2
import os

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("EmotionDetectionPreprocessing") \
    .getOrCreate()

def preprocess_image_spark(pixels_str):
    """
    UDF for preprocessing FER2013 pixel strings in parallel.
    Resizes and normalizes the image.
    """
    try:
        # Convert string to numpy array
        face = np.fromstring(pixels_str, sep=' ', dtype=np.uint8).reshape(48, 48)
        # Resize to 64x64 for ResNet
        face_resized = cv2.resize(face, (64, 64))
        # Normalize
        face_normalized = face_resized.astype('float32') / 255.0
        return face_normalized.flatten().tolist()
    except:
        return None

# Register UDF
preprocess_udf = udf(preprocess_image_spark, ArrayType(FloatType()))

def main(input_path='fer2013.csv', output_path='preprocessed_data_spark'):
    # Load dataset (HDFS or Local)
    print(f"Loading data from {input_path}...")
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    
    # Filter and apply preprocessing
    print("Running distributed preprocessing...")
    processed_df = df.withColumn("processed_pixels", preprocess_udf(col("pixels")))
    
    # Filter out failures
    processed_df = processed_df.filter(col("processed_pixels").isNotNull())
    
    # Save results (Parquet is better for large scale data science)
    print(f"Saving preprocessed data to {output_path}...")
    processed_df.select("emotion", "processed_pixels", "Usage") \
        .write.mode("overwrite").parquet(output_path)
    
    print("Spark job completed successfully.")

if __name__ == "__main__":
    # Note: Ensure spark-submit is used or pyspark is installed
    # Example: spark-submit spark_preprocess.py
    main()
