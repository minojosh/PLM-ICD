#!/usr/bin/env python3
"""
Extract unique ICD codes from ALL_CODES.csv to create ALL_CODES.txt

This script converts the CAML-format ALL_CODES.csv file to the PLM-format ALL_CODES.txt
file required for training. It extracts all unique ICD-10 codes and writes them one per line.

Usage:
    python utils/create_codes_file.py

Input:  data/mimic4/mimic4_icd10/ALL_CODES.csv
Output: data/mimic4/mimic4_icd10/ALL_CODES.txt
"""

import pandas as pd
import os

def create_codes_file(input_csv_path, output_txt_path):
    """
    Extract unique ICD codes from CSV and write to text file.
    
    Args:
        input_csv_path (str): Path to ALL_CODES.csv file
        output_txt_path (str): Path to output ALL_CODES.txt file
    """
    print(f"Reading ICD codes from: {input_csv_path}")
    
    # Read the ALL_CODES.csv file
    df = pd.read_csv(input_csv_path, dtype={"ICD10_CODE": str})
    
    # Extract unique ICD codes and sort them
    unique_codes = sorted(df['ICD10_CODE'].unique())
    
    print(f"Found {len(unique_codes)} unique ICD-10 codes")
    
    # Write to ALL_CODES.txt (one code per line)
    with open(output_txt_path, 'w') as f:
        for code in unique_codes:
            f.write(f"{code}\n")
    
    print(f"Created {output_txt_path} successfully!")
    return len(unique_codes)

def main():
    # Define paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(project_root, 'data/mimic4/mimic4_icd10/ALL_CODES.csv')
    output_txt = os.path.join(project_root, 'data/mimic4/mimic4_icd10/ALL_CODES.txt')
    
    # Check if input file exists
    if not os.path.exists(input_csv):
        print(f"Error: Input file not found: {input_csv}")
        print("Please ensure the CAML data processing has been completed.")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    
    # Convert the file
    num_codes = create_codes_file(input_csv, output_txt)
    
    print(f"\nConversion completed successfully!")
    print(f"  Input:  {input_csv}")
    print(f"  Output: {output_txt}")
    print(f"  Codes:  {num_codes} unique ICD-10 codes")
    
    return 0

if __name__ == "__main__":
    exit(main())