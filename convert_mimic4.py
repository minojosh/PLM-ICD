#!/usr/bin/env python3
"""
Convert MIMIC-4 JSON data to CSV format compatible with PLM-ICD training.
"""

import json
import pandas as pd
import argparse
from pathlib import Path

def convert_mimic4_to_csv(json_file, output_dir, train_ratio=0.7, val_ratio=0.15):
    """Convert MIMIC-4 JSON to train/dev/test CSV files."""
    
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    records = data['records']
    print(f"Loaded {len(records)} records from {json_file}")
    
    # Convert to DataFrame
    processed_records = []
    for record in records:
        processed_record = {
            'HADM_ID': record['hadm_id'],
            'SUBJECT_ID': record['subject_id'], 
            'TEXT': record['note_text'],
            'LABELS': record['icd_codes']  # CSV format with comma-separated codes
        }
        processed_records.append(processed_record)
    
    df = pd.DataFrame(processed_records)
    
    # Split data
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Shuffle and split
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df = df_shuffled[:n_train]
    val_df = df_shuffled[n_train:n_train+n_val]
    test_df = df_shuffled[n_train+n_val:]
    
    # Save to CSV
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    train_df.to_csv(output_path / 'train_full.csv', index=False)
    val_df.to_csv(output_path / 'dev_full.csv', index=False)
    test_df.to_csv(output_path / 'test_full.csv', index=False)
    
    print(f"Saved {len(train_df)} train, {len(val_df)} dev, {len(test_df)} test records to {output_dir}")
    
    # Extract unique codes for ALL_CODES.txt
    all_codes = set()
    for record in records:
        codes = record['icd_codes'].split(',')
        all_codes.update(codes)
    
    with open(output_path / 'ALL_CODES.txt', 'w') as f:
        for code in sorted(all_codes):
            f.write(f"{code}\n")
    
    print(f"Extracted {len(all_codes)} unique ICD codes to ALL_CODES.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input MIMIC-4 JSON file")
    parser.add_argument("--output", required=True, help="Output directory for CSV files")
    args = parser.parse_args()
    
    convert_mimic4_to_csv(args.input, args.output)