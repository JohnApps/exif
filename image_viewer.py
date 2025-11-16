# python image_viewer.py
import duckdb
from PIL import Image, ImageTk # Pillow library for image processing
import time
import os
import tkinter as tk # For actual image display (optional, but good to know)

# --- Configuration ---
DATABASE_FILE = 'exif_db.db'
TABLE_NAME = 'exif' # **IMPORTANT: Replace with your actual table name**
IMAGE_DISPLAY_INTERVAL_SECONDS = 60

def display_images_from_db(db_file, table_name, interval):
    """
    Connects to the DuckDB database, fetches image file paths,
    and "displays" them with a specified interval.
    """
    try:
        # Connect to DuckDB
        con = duckdb.connect(database=db_file, read_only=True)
        print(f"Successfully connected to DuckDB database: {db_file}")

        # Fetch image file paths from the database
        query = f"SELECT filepath, filename FROM {table_name} WHERE filepath IS NOT NULL AND filepath != '' ORDER BY createdate, createtime;"
        result = con.execute(query).fetchall()

        if not result:
            print("No image file paths found in the database.")
            return

        print(f"Found {len(result)} image entries.")

        # --- FIX START ---
        # Initialize root and label to None
        root = None
        label = None
        # --- FIX END ---

        # Optional: For actual GUI display
        # You can change 'True' to a configurable variable if you want to easily
        # switch between GUI and non-GUI mode.
        if True: # Set to False if you don't want a Tkinter window
            root = tk.Tk()
            root.title("Image Slideshow")
            label = tk.Label(root)
            label.pack()

        for i, (filepath, filename) in enumerate(result):
            print(f"\n--- Displaying Image {i+1}/{len(result)} ---")
            full_path = filepath # Assuming filepath is already a full path

            if not os.path.exists(full_path):
                print(f"Warning: Image file not found at path: {full_path}")
                continue

            try:
                # Open the image using Pillow
                img = Image.open(full_path)
                print(f"Processing: {filename} (Path: {full_path})")
                print(f"Image dimensions: {img.width}x{img.height}")

                # This block now correctly checks if root and label were initialized
                if root and label:
                    # Resize image to fit window (optional)
                    img.thumbnail((800, 600), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    label.config(image=photo)
                    label.image = photo # Keep a reference!
                    root.update_idletasks() # Update the display
                    root.update() # Process events

                # Simulate "display" by holding for the interval
                print(f"Waiting for {interval} seconds...")
                time.sleep(interval)

            except IOError:
                print(f"Error: Could not open or process image file: {full_path}")
            except Exception as e:
                print(f"An unexpected error occurred with {full_path}: {e}")

        print("\n--- All images processed ---")

    except duckdb.Error as e:
        print(f"DuckDB Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'con' in locals() and con:
            con.close()
            print("DuckDB connection closed.")
        # This check is now safe because root is always initialized
        if root:
           root.destroy() # Close the Tkinter window
# --- Main execution ---
if __name__ == "__main__":
    # Run the image display function
    display_images_from_db(DATABASE_FILE, TABLE_NAME, IMAGE_DISPLAY_INTERVAL_SECONDS)    # Create a dummy DuckDB database and table for testing if it doesn't exist
    # In a real scenario, your database and table would already be populated.
    try:
        con_test = duckdb.connect(database=DATABASE_FILE)
        # Create the table if it doesn't exist
        con_test.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id BIGINT DEFAULT(nextval('exif_id_seq')),
                filepath VARCHAR,
                filename VARCHAR,
                createdate DATE,
                createtime TIME,
                camera_make VARCHAR,
                camera_model VARCHAR,
                exposure_time VARCHAR,
                aperture VARCHAR,
                iso VARCHAR,
                content_tags VARCHAR[]
            );
        """)
        # Create a sequence for the id if it doesn't exist
        con_test.execute("CREATE SEQUENCE IF NOT EXISTS exif_id_seq;")

        # Insert some dummy data if the table is empty
        # **IMPORTANT: Replace with paths to actual image files on your system for testing**
        if con_test.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};").fetchone()[0] == 0:
            print("Inserting dummy data for demonstration. Please update with your actual image paths.")
            dummy_image_path1 = "o:\\bilder\\JohnSmall.JPG" # Example Windows path
            dummy_image_path2 = "o:\\bilder\\JohnSmall.JPG" # Example Windows path
            # For Linux/macOS, use paths like "/home/user/Pictures/image1.jpg"

            # Check if dummy paths exist, if not, try common sample paths or skip
            if not os.path.exists(dummy_image_path1):
                print(f"Warning: Dummy image 1 not found at {dummy_image_path1}. Skipping insertion for this image.")
                dummy_image_path1 = None
            if not os.path.exists(dummy_image_path2):
                print(f"Warning: Dummy image 2 not found at {dummy_image_path2}. Skipping insertion for this image.")
                dummy_image_path2 = None

            if dummy_image_path1:
                con_test.execute(f"""
                    INSERT INTO {TABLE_NAME} (filepath, filename, createdate, createtime) VALUES
                    (?, ?, ?, ?);
                """, [dummy_image_path1, os.path.basename(dummy_image_path1), '2023-01-01', '10:00:00'])
            if dummy_image_path2:
                con_test.execute(f"""
                    INSERT INTO {TABLE_NAME} (filepath, filename, createdate, createtime) VALUES
                    (?, ?, ?, ?);
                """, [dummy_image_path2, os.path.basename(dummy_image_path2), '2023-01-02', '11:30:00'])

            con_test.commit()
            print("Dummy data inserted.")
        con_test.close()

    except Exception as e:
        print(f"Error setting up dummy database/table: {e}")
        print("Please ensure you have DuckDB installed and write access to the directory.")

    # Run the image display function
    display_images_from_db(DATABASE_FILE, TABLE_NAME, IMAGE_DISPLAY_INTERVAL_SECONDS)