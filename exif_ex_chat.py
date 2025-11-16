# exif_ex_chat.py
#
import duckdb
import concurrent.futures
import os
from PIL import Image
from transformers import pipeline
import pandas as pd

# Prevent DOS bomb image attacks
Image.MAX_IMAGE_PIXELS = None

# Connect to DuckDB with memory and thread configuration
con = duckdb.connect(database="chat_exif_data.db", read_only=False)
con.execute("SET memory_limit='16GB'")
con.execute("SET threads=16")

# Fetch JPG Nikon images
query = """
    SELECT id, filepath 
    FROM exif_data 
    WHERE file_extension = 'JPG' 
      AND lower(camera_make) LIKE '%nikon%'
"""
df = con.execute(query).fetchdf()

# Load content classification pipeline (e.g., MobileNet, ResNet, etc.)
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

# Labels we're interested in
TARGET_LABELS = {
    "landscape", "bird", "lake", "ocean", "building", "person", "tree", 
    "road", "food", "drink", "wind turbine", "windmill", "wind energy"
}

# Function to classify one image
def classify_image(row):
    image_id, filepath = row
    try:
        if not os.path.exists(filepath):
            return image_id, filepath, []
        preds = classifier(Image.open(filepath))
        labels = {p['label'].lower() for p in preds}
        detected = list(label for label in labels if any(t in label for t in TARGET_LABELS))
        return image_id, filepath, detected
    except Exception as e:
        return image_id, filepath, [f"error: {str(e)}"]

# Run classification with 4 threads
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = executor.map(classify_image, df.itertuples(index=False, name=None))
    results = list(futures)

# Prepare results DataFrame
results_df = pd.DataFrame(results, columns=["id", "filepath", "detected_content"])
results_df["detected_content"] = results_df["detected_content"].apply(lambda x: ', '.join(x))

# Write results to a new table
con.execute("DROP TABLE IF EXISTS image_content_analysis")
con.register("results_df", results_df)
con.execute("""
    CREATE TABLE image_content_analysis AS 
    SELECT id, filepath, detected_content 
    FROM results_df
""")

# Show DuckDB version
duckdb_version = con.execute("SELECT version()").fetchone()[0]
print("DuckDB Version:", duckdb_version)

# Close the connection
con.close()
