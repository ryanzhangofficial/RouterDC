#!/usr/bin/env python3
import os
import json
import pandas as pd
import glob
from pathlib import Path

def process_folder(folder_path):
    """Process a single folder: combine CSVs, convert to JSON format, cleanup CSVs"""
    folder_name = os.path.basename(folder_path)
    print(f"\nProcessing folder: {folder_name}")
    
    # Find all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print(f"  No CSV files found in {folder_name}, skipping...")
        return
    
    print(f"  Found {len(csv_files)} CSV files")
    
    # Combine all CSV files
    combined_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            combined_data.append(df)
            print(f"    Loaded {csv_file} ({len(df)} rows)")
        except Exception as e:
            print(f"    Error loading {csv_file}: {e}")
            continue
    
    if not combined_data:
        print(f"  No valid CSV data found in {folder_name}")
        return
    
    # Concatenate all dataframes
    combined_df = pd.concat(combined_data, ignore_index=True)
    print(f"  Combined dataset: {len(combined_df)} total rows")
    
    # Convert to the required JSON format
    converted_data = []
    doc_id = 0
    
    for _, row in combined_df.iterrows():
        try:
            # Create the converted object
            converted_obj = {
                "doc_id": doc_id,
                "question": f"Question: {row.get('input_text', row.get('question', ''))}",
                "scores": {}
            }
            
            # Convert score keys from label_xsmall -> xsmall, etc.
            for col, value in row.items():
                if col.startswith('label_') or col in ['xsmall', 'small', 'medium', 'large']:
                    # Remove label_ prefix if present
                    score_key = col.replace('label_', '')
                    if score_key in ['xsmall', 'small', 'medium', 'large']:
                        # Convert to float, handle NaN values
                        try:
                            score_value = float(value) if pd.notna(value) else 0.0
                            converted_obj["scores"][score_key] = score_value
                        except (ValueError, TypeError):
                            converted_obj["scores"][score_key] = 0.0
            
            # Ensure we have all required score keys
            for size in ['xsmall', 'small', 'medium', 'large']:
                if size not in converted_obj["scores"]:
                    converted_obj["scores"][size] = 0.0
            
            # Only add if we have a valid question
            if converted_obj["question"].strip() != "Question: ":
                converted_data.append(converted_obj)
                doc_id += 1
                
        except Exception as e:
            print(f"    Error processing row {doc_id}: {e}")
            continue
    
    print(f"  Converted {len(converted_data)} valid entries")
    
    if not converted_data:
        print(f"  No valid data to save for {folder_name}")
        return
    
    # Save as JSON file
    json_filename = f"{folder_name}_qwen.json"
    json_filepath = os.path.join(folder_path, json_filename)
    
    try:
        with open(json_filepath, 'w') as f:
            json.dump(converted_data, f, indent=2)
        print(f"  Saved JSON: {json_filepath}")
        
        # Show sample of converted data
        print(f"  Sample entry:")
        print(f"    Question: {converted_data[0]['question'][:100]}...")
        print(f"    Scores: {converted_data[0]['scores']}")
        
        # Delete CSV files after successful JSON creation
        print(f"  Cleaning up CSV files...")
        for csv_file in csv_files:
            try:
                os.remove(csv_file)
                print(f"    Deleted: {os.path.basename(csv_file)}")
            except Exception as e:
                print(f"    Error deleting {csv_file}: {e}")
        
        print(f"  ✅ Successfully processed {folder_name}")
        
    except Exception as e:
        print(f"  ❌ Error saving JSON for {folder_name}: {e}")

def main():
    """Main function to process all folders"""
    base_path = "/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/data/qwen2_narrow_cost_spread"
    
    print(f"Processing folders in: {base_path}")
    
    if not os.path.exists(base_path):
        print(f"❌ Base path does not exist: {base_path}")
        return
    
    # Get all subdirectories
    folders = [f for f in os.listdir(base_path) 
              if os.path.isdir(os.path.join(base_path, f))]
    
    if not folders:
        print("No folders found to process")
        return
    
    print(f"Found {len(folders)} folders to process:")
    for folder in sorted(folders):
        print(f"  - {folder}")
    
    print("\n" + "="*60)
    print("Starting processing...")
    
    # Process each folder
    for folder in sorted(folders):
        folder_path = os.path.join(base_path, folder)
        try:
            process_folder(folder_path)
        except Exception as e:
            print(f"❌ Error processing folder {folder}: {e}")
            continue
    
    print("\n" + "="*60)
    print("✅ Processing complete!")
    
    # Summary
    json_files = []
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        json_file = os.path.join(folder_path, f"{folder}_qwen.json")
        if os.path.exists(json_file):
            json_files.append(json_file)
    
    print(f"\nSummary:")
    print(f"  Processed folders: {len(folders)}")
    print(f"  JSON files created: {len(json_files)}")
    
    if json_files:
        print(f"\nCreated JSON files:")
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    print(f"  - {os.path.basename(json_file)}: {len(data)} entries")
            except:
                print(f"  - {os.path.basename(json_file)}: Error reading file")

if __name__ == "__main__":
    main()