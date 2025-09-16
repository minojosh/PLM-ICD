#!/usr/bin/env python3
"""
Convert CAML-processed MIMIC-4 data to PLM-ICD format.

CAML format: subject_id,hadm_id,text,labels,length
PLM format:  text,label (only text and label columns needed for training)
"""

import pandas as pd
import os
from collections import Counter

def convert_caml_to_plm(caml_file, plm_file):
    """Convert a single CAML CSV file to PLM format."""
    print(f"Converting {caml_file} to {plm_file}")
    
    # Read CAML format
    df = pd.read_csv(caml_file, encoding='utf-8')
    print(f"Loaded {len(df)} samples from {caml_file}")
    
    # Convert to PLM format - only need text and label columns
    plm_df = pd.DataFrame({
        'text': df['text'],
        'label': df['labels']
    })
    
    # Save PLM format
    plm_df.to_csv(plm_file, index=False, encoding='utf-8')
    print(f"Saved {len(plm_df)} samples to {plm_file}")
    
    return plm_df

def extract_all_codes(train_df, dev_df, test_df, output_file):
    """Extract all unique ICD codes from the datasets."""
    all_labels = []
    
    for df in [train_df, dev_df, test_df]:
        for labels_str in df['label']:
            if pd.notna(labels_str):
                codes = labels_str.split(';')
                all_labels.extend(codes)
    
    unique_codes = sorted(set(all_labels))
    print(f"Found {len(unique_codes)} unique ICD codes")
    
    # Save codes to file (one per line, no header)
    with open(output_file, 'w') as f:
        for code in unique_codes:
            f.write(f"{code}\n")
    
    print(f"Saved unique codes to {output_file}")
    return unique_codes

def main():
    # Source directory with CAML data
    caml_dir = "data/mimic4/mimic4_icd10"
    
    # Target directory for PLM data
    plm_dir = "data/mimic4"
    os.makedirs(plm_dir, exist_ok=True)
    
    # File mappings
    file_mappings = {
        "train_full.csv": "train_full.csv",
        "dev_full.csv": "dev_full.csv", 
        "test_full.csv": "test_full.csv"
    }
    
    converted_dfs = {}
    
    # Convert each file
    for caml_file, plm_file in file_mappings.items():
        caml_path = os.path.join(caml_dir, caml_file)
        plm_path = os.path.join(plm_dir, plm_file)
        
        if os.path.exists(caml_path):
            converted_df = convert_caml_to_plm(caml_path, plm_path)
            converted_dfs[plm_file] = converted_df
        else:
            print(f"Warning: {caml_path} not found")
    
    # Extract all unique codes
    if converted_dfs:
        codes_file = os.path.join(plm_dir, "ALL_CODES.txt")
        extract_all_codes(
            converted_dfs.get("train_full.csv", pd.DataFrame()),
            converted_dfs.get("dev_full.csv", pd.DataFrame()),
            converted_dfs.get("test_full.csv", pd.DataFrame()),
            codes_file
        )
    
    print("\nConversion completed!")
    print(f"PLM format files are in: {plm_dir}")
    print("Files created:")
    for plm_file in file_mappings.values():
        plm_path = os.path.join(plm_dir, plm_file)
        if os.path.exists(plm_path):
            print(f"  - {plm_path}")
    
    codes_file = os.path.join(plm_dir, "ALL_CODES.txt")
    if os.path.exists(codes_file):
        print(f"  - {codes_file}")

if __name__ == "__main__":
    main()