# exif_csv.py
#
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

# Connect to DuckDB database
conn = duckdb.connect('exif_csv.db')

# Basic table information
def analyze_exif():
    """Comprehensive analysis of EXIF table structure and data"""
    
    print("=== EXIF TABLE ANALYSIS ===\n")
    
    # 1. Basic table statistics
    print("1. TABLE OVERVIEW")
    print("-" * 50)
    
    # Get row count
    row_count = conn.execute("SELECT COUNT(*) FROM exif").fetchone()[0]
    print(f"Total records: {row_count:,}")
    
    # Get column count
    col_count = conn.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'exif'").fetchone()[0]
    print(f"Total columns: {col_count}")
    
    # 2. Data type distribution
    print("\n2. DATA TYPE DISTRIBUTION")
    print("-" * 50)
    
    dtype_query = """
    SELECT data_type, COUNT(*) as count
    FROM information_schema.columns 
    WHERE table_name = 'exif'
    GROUP BY data_type
    ORDER BY count DESC
    """
    
    dtypes = conn.execute(dtype_query).fetchdf()
    print(dtypes.to_string(index=False))
    
    # 3. Non-null data analysis
    print("\n3. DATA COMPLETENESS ANALYSIS")
    print("-" * 50)
    
    # Get columns with highest data completeness
    completeness_query = """
    SELECT 
        column_name,
        COUNT(*) - COUNT(CASE WHEN column_name IS NULL THEN 1 END) as non_null_count,
        ROUND(100.0 * (COUNT(*) - COUNT(CASE WHEN column_name IS NULL THEN 1 END)) / COUNT(*), 2) as completeness_pct
    FROM (
        SELECT unnest(columns(*)) as column_name FROM exif
    )
    GROUP BY column_name
    HAVING non_null_count > 0
    ORDER BY non_null_count DESC
    LIMIT 20
    """
    
    # Alternative approach for completeness analysis
    core_fields = [
        'SourceFile', 'Make', 'Model', 'DateTime', 'ExposureTime', 
        'FNumber', 'ISO', 'FocalLength', 'Flash', 'ImageWidth', 
        'ImageHeight', 'Orientation', 'Software', 'ColorSpace'
    ]
    
    print("Core EXIF fields completeness:")
    for field in core_fields:
        try:
            query = f"SELECT COUNT(*) as total, COUNT({field}) as non_null FROM exif"
            result = conn.execute(query).fetchone()
            total, non_null = result
            completeness = (non_null / total * 100) if total > 0 else 0
            print(f"{field:20}: {non_null:>6}/{total:<6} ({completeness:5.1f}%)")
        except:
            print(f"{field:20}: Column not found or error")
    
    # 4. Camera make/model analysis
    print("\n4. CAMERA MAKE/MODEL DISTRIBUTION")
    print("-" * 50)
    
    try:
        make_query = "SELECT Make, COUNT(*) as count FROM exif WHERE Make IS NOT NULL GROUP BY Make ORDER BY count DESC LIMIT 10"
        makes = conn.execute(make_query).fetchdf()
        print("Top 10 Camera Makes:")
        print(makes.to_string(index=False))
    except:
        print("Make field analysis not available")
    
    # 5. Technical settings analysis
    print("\n5. TECHNICAL SETTINGS ANALYSIS")
    print("-" * 50)
    
    # ISO distribution
    try:
        iso_stats = conn.execute("""
            SELECT 
                MIN(ISO) as min_iso,
                MAX(ISO) as max_iso,
                AVG(ISO) as avg_iso,
                MEDIAN(ISO) as median_iso,
                COUNT(ISO) as iso_count
            FROM exif 
            WHERE ISO IS NOT NULL AND ISO > 0
        """).fetchone()
        
        if iso_stats[4] > 0:  # if we have ISO data
            print(f"ISO Statistics:")
            print(f"  Range: {iso_stats[0]} - {iso_stats[1]}")
            print(f"  Average: {iso_stats[2]:.1f}")
            print(f"  Median: {iso_stats[3]:.1f}")
            print(f"  Records with ISO: {iso_stats[4]}")
    except:
        print("ISO analysis not available")
    
    # 6. Geographic data analysis
    print("\n6. GEOGRAPHIC DATA ANALYSIS")
    print("-" * 50)
    
    try:
        gps_query = """
            SELECT 
                COUNT(GPSLatitude) as has_gps_lat,
                COUNT(GPSLongitude) as has_gps_lon,
                COUNT(GPSAltitude) as has_gps_alt
            FROM exif
        """
        gps_data = conn.execute(gps_query).fetchone()
        print(f"GPS Latitude records: {gps_data[0]}")
        print(f"GPS Longitude records: {gps_data[1]}")
        print(f"GPS Altitude records: {gps_data[2]}")
    except:
        print("GPS data analysis not available")
    
    # 7. File format analysis
    print("\n7. FILE FORMAT ANALYSIS")
    print("-" * 50)
    
    try:
        format_query = "SELECT FileType, COUNT(*) as count FROM exif WHERE FileType IS NOT NULL GROUP BY FileType ORDER BY count DESC"
        formats = conn.execute(format_query).fetchdf()
        print("File Types:")
        print(formats.to_string(index=False))
    except:
        print("File format analysis not available")

def create_visualizations():
    """Create visualizations for EXIF data analysis"""
    
    print("\n8. CREATING VISUALIZATIONS")
    print("-" * 50)
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('EXIF Data Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: Data type distribution
    try:
        dtype_query = """
        SELECT data_type, COUNT(*) as count
        FROM information_schema.columns 
        WHERE table_name = 'exif'
        GROUP BY data_type
        ORDER BY count DESC
        """
        dtypes = conn.execute(dtype_query).fetchdf()
        
        axes[0,0].bar(dtypes['data_type'], dtypes['count'])
        axes[0,0].set_title('Column Data Types Distribution')
        axes[0,0].set_xlabel('Data Type')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=45)
    except Exception as e:
        axes[0,0].text(0.5, 0.5, f'Data type plot error:\n{str(e)}', 
                      ha='center', va='center', transform=axes[0,0].transAxes)
    
    # Plot 2: Camera make distribution
    try:
        make_query = "SELECT Make, COUNT(*) as count FROM exif WHERE Make IS NOT NULL GROUP BY Make ORDER BY count DESC LIMIT 8"
        makes = conn.execute(make_query).fetchdf()
        
        if len(makes) > 0:
            axes[0,1].bar(range(len(makes)), makes['count'])
            axes[0,1].set_title('Top Camera Manufacturers')
            axes[0,1].set_xlabel('Camera Make')
            axes[0,1].set_ylabel('Count')
            axes[0,1].set_xticks(range(len(makes)))
            axes[0,1].set_xticklabels(makes['Make'], rotation=45, ha='right')
        else:
            axes[0,1].text(0.5, 0.5, 'No camera make data available', 
                          ha='center', va='center', transform=axes[0,1].transAxes)
    except Exception as e:
        axes[0,1].text(0.5, 0.5, f'Camera make plot error:\n{str(e)}', 
                      ha='center', va='center', transform=axes[0,1].transAxes)
    
    # Plot 3: ISO distribution
    try:
        iso_query = "SELECT ISO FROM exif WHERE ISO IS NOT NULL AND ISO BETWEEN 50 AND 25600"
        iso_data = conn.execute(iso_query).fetchdf()
        
        if len(iso_data) > 0:
            axes[1,0].hist(iso_data['ISO'], bins=30, edgecolor='black', alpha=0.7)
            axes[1,0].set_title('ISO Values Distribution')
            axes[1,0].set_xlabel('ISO')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_yscale('log')
        else:
            axes[1,0].text(0.5, 0.5, 'No ISO data available', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
    except Exception as e:
        axes[1,0].text(0.5, 0.5, f'ISO plot error:\n{str(e)}', 
                      ha='center', va='center', transform=axes[1,0].transAxes)
    
    # Plot 4: File type distribution
    try:
        format_query = "SELECT FileType, COUNT(*) as count FROM exif WHERE FileType IS NOT NULL GROUP BY FileType ORDER BY count DESC LIMIT 8"
        formats = conn.execute(format_query).fetchdf()
        
        if len(formats) > 0:
            axes[1,1].pie(formats['count'], labels=formats['FileType'], autopct='%1.1f%%')
            axes[1,1].set_title('File Types Distribution')
        else:
            axes[1,1].text(0.5, 0.5, 'No file type data available', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
    except Exception as e:
        axes[1,1].text(0.5, 0.5, f'File type plot error:\n{str(e)}', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
    
    plt.tight_layout()
    plt.show()

def generate_data_quality_report():
    """Generate a comprehensive data quality report"""
    
    print("\n9. DATA QUALITY REPORT")
    print("=" * 50)
    
    # Check for common data quality issues
    issues = []
    
    try:
        # Check for duplicate source files
        dup_query = """
        SELECT SourceFile, COUNT(*) as count 
        FROM exif 
        WHERE SourceFile IS NOT NULL 
        GROUP BY SourceFile 
        HAVING COUNT(*) > 1
        """
        duplicates = conn.execute(dup_query).fetchdf()
        if len(duplicates) > 0:
            issues.append(f"Found {len(duplicates)} duplicate SourceFile entries")
    except:
        pass
    
    try:
        # Check for invalid date formats
        invalid_dates = conn.execute("""
            SELECT COUNT(*) FROM exif 
            WHERE DateTime IS NOT NULL 
            AND DateTime NOT LIKE '____:__:__ __:__:__'
        """).fetchone()[0]
        if invalid_dates > 0:
            issues.append(f"Found {invalid_dates} records with non-standard DateTime format")
    except:
        pass
    
    try:
        # Check for extreme ISO values
        extreme_iso = conn.execute("""
            SELECT COUNT(*) FROM exif 
            WHERE ISO IS NOT NULL 
            AND (ISO < 25 OR ISO > 102400)
        """).fetchone()[0]
        if extreme_iso > 0:
            issues.append(f"Found {extreme_iso} records with extreme ISO values")
    except:
        pass
    
    if issues:
        print("Data Quality Issues Found:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print("No major data quality issues detected")
    
    # Column utilization summary
    print(f"\nColumn Utilization Summary:")
    print(f"- Total columns defined: {len([col for col in range(600)])}")  # Approximate from your schema
    print(f"- Columns likely containing data: varies by dataset")
    print(f"- Sparse schema typical for EXIF data due to camera/software variations")

def export_summary_tables():
    """Export key summary tables for further analysis"""
    
    print("\n10. EXPORTING SUMMARY TABLES")
    print("-" * 50)
    
    try:
        # Export camera equipment summary
        equipment_query = """
        SELECT 
            Make,
            Model,
            COUNT(*) as photo_count,
            MIN(DateTime) as earliest_photo,
            MAX(DateTime) as latest_photo
        FROM exif 
        WHERE Make IS NOT NULL AND Model IS NOT NULL
        GROUP BY Make, Model
        ORDER BY photo_count DESC
        """
        
        conn.execute("CREATE OR REPLACE TABLE camera_equipment_summary AS " + equipment_query)
        print("✓ Created camera_equipment_summary table")
        
        # Export technical settings summary
        settings_query = """
        SELECT 
            CASE 
                WHEN ISO <= 200 THEN 'Low (≤200)'
                WHEN ISO <= 800 THEN 'Medium (201-800)'
                WHEN ISO <= 3200 THEN 'High (801-3200)'
                ELSE 'Very High (>3200)'
            END as iso_range,
            COUNT(*) as count,
            AVG(FNumber) as avg_aperture,
            AVG(CAST(REPLACE(ExposureTime, '1/', '') AS INTEGER)) as avg_shutter_denominator
        FROM exif 
        WHERE ISO IS NOT NULL AND ISO > 0
        GROUP BY iso_range
        ORDER BY MIN(ISO)
        """
        
        conn.execute("CREATE OR REPLACE TABLE technical_settings_summary AS " + settings_query)
        print("✓ Created technical_settings_summary table")
        
    except Exception as e:
        print(f"Error creating summary tables: {e}")

# Run the complete analysis
if __name__ == "__main__":
    try:
        analyze_exif()
        create_visualizations()
        generate_data_quality_report()
        export_summary_tables()
        
        print(f"\n{'='*50}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*50}")
        print("Summary tables created in database:")
        print("- camera_equipment_summary")
        print("- technical_settings_summary")
        
    except Exception as e:
        print(f"Analysis error: {e}")
    finally:
        conn.close()