from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, array
from pyspark.sql.types import StructType, StructField, ArrayType, FloatType, StringType
import numpy as np
import cv2
import os

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("EmotionDetectionPreprocessing") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

def apply_synthetic_mask(face_img):
    """
    Applies a simple geometric synthetic surgical mask to the lower half of the face.
    Operates on a 64x64 numpy array [0-1].
    """
    masked_img = face_img.copy()
    
    # 64x64 image coordinates
    # Mask covers roughly the bottom half (y > 32)
    # Give it some curves to look like a surgical mask
    pts = np.array([[10, 32], [54, 32], [60, 64], [4, 64]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    # Draw white polygon (value 1.0 since it's normalized)
    cv2.fillPoly(masked_img, [pts], 1.0)
    
    return masked_img

def process_and_mask_spark(pixels_str):
    """
    Spark UDF: Takes FER2013 string -> Resizes to 64x64 -> Generates Target & Masked Pairs
    """
    try:
        # 1. Parsing the string and reshaping to original FER2013 48x48
        face = np.fromstring(pixels_str, sep=' ', dtype=np.uint8).reshape(48, 48)
        
        # 2. Resizing to 64x64 (Target size for ResNet/Autoencoder)
        face_resized = cv2.resize(face, (64, 64))
        
        # 3. Normalization [0, 1]
        face_normalized = face_resized.astype('float32') / 255.0
        
        # 4. Generate the Masked Version
        face_masked = apply_synthetic_mask(face_normalized)
        
        # Return a dictionary/struct format expected by the UDF return type
        return {
            "original_pixels": face_normalized.flatten().tolist(),
            "masked_pixels": face_masked.flatten().tolist()
        }
    except Exception as e:
        return None

# Define the schema for the struct returned by the UDF
schema = StructType([
    StructField("original_pixels", ArrayType(FloatType()), False),
    StructField("masked_pixels", ArrayType(FloatType()), False)
])

# Register UDF
process_udf = udf(process_and_mask_spark, schema)

def main(input_path='fer2013.csv', output_path='preprocessed_data.parquet'):
    if not os.path.exists(input_path):
        print(f"ERROR: Cannot find {input_path}. Please download FER2013 dataset first.")
        return
        
    print(f"Loading data from {input_path}...")
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    
    print("Running distributed preprocessing and synthetic mask generation...")
    # Apply the UDF
    processed_df = df.withColumn("processed_struct", process_udf(col("pixels")))
    
    # Filter failures
    processed_df = processed_df.filter(col("processed_struct").isNotNull())
    
    # Unpack the struct into separate columns
    final_df = processed_df.select(
        col("emotion").cast("int"),
        col("processed_struct.masked_pixels").alias("masked_pixels"),
        col("processed_struct.original_pixels").alias("original_pixels"),
        col("Usage")
    )
    
    # Save the polished dataset directly to Parquet
    print(f"Saving distributed dataset to {output_path}...")
    final_df.write.mode("overwrite").parquet(output_path)
    
    print("Spark Pipeline Completed Successfully! Deep Learning team can now read from:", output_path)

if __name__ == "__main__":
    main()
