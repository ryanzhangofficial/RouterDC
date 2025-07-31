#!/usr/bin/env python3
import os
import json
import pandas as pd
import glob
from pathlib import Path

def convert_json_format(input_json_path, output_json_path):
    """Convert JSON from qwen2_narrow_cost_spread format to inference_outputs format"""
    try:
        print(f"  Converting {input_json_path}...")
        
        # Load the input JSON
        with open(input_json_path, 'r') as f:
            input_data = json.load(f)
        
        print(f"    Loaded {len(input_data)} entries")
        
        # Convert to target format
        converted_data = []
        
        for entry in input_data:
            try:
                # Create converted entry
                converted_entry = {
                    "doc_id": int(entry.get("doc_id", 0)),
                    "question": f"Question: {entry.get('input_text', '')}",
                    "scores": {}
                }
                
                # Convert label_* fields to scores object
                for size in ['xsmall', 'small', 'medium', 'large']:
                    label_key = f"label_{size}"
                    if label_key in entry:
                        # Handle NaN values and convert to float
                        try:
                            score_value = float(entry[label_key]) if pd.notna(entry[label_key]) else 0.0
                            converted_entry["scores"][size] = score_value
                        except (ValueError, TypeError):
                            converted_entry["scores"][size] = 0.0
                    else:
                        converted_entry["scores"][size] = 0.0
                
                # Only add if we have a valid question
                if converted_entry["question"].strip() != "Question: ":
                    converted_data.append(converted_entry)
                    
            except Exception as e:
                print(f"    Error processing entry with doc_id {entry.get('doc_id', 'unknown')}: {e}")
                continue
        
        print(f"    Converted {len(converted_data)} valid entries")
        
        if not converted_data:
            print(f"    No valid data to save")
            return False
        
        # Create backup before overwriting (if replacing original file)
        if input_json_path == output_json_path:
            backup_path = input_json_path + ".backup"
            print(f"    Creating backup: {backup_path}")
            import shutil
            shutil.copy2(input_json_path, backup_path)
        
        # Save converted data (overwrites original if paths are the same)
        with open(output_json_path, 'w') as f:
            json.dump(converted_data, f, indent=2)
        
        print(f"    Saved converted JSON: {output_json_path}")
        
        # Show sample of converted data
        print(f"    Sample entry:")
        print(f"      Doc ID: {converted_data[0]['doc_id']}")
        print(f"      Question: {converted_data[0]['question'][:100]}...")
        print(f"      Scores: {converted_data[0]['scores']}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Error converting {input_json_path}: {e}")
        # If we created a backup and conversion failed, restore it
        if input_json_path == output_json_path:
            backup_path = input_json_path + ".backup"
            if os.path.exists(backup_path):
                print(f"    Restoring backup due to error...")
                import shutil
                shutil.copy2(backup_path, input_json_path)
        return False

def process_folder(folder_path):
    """Process a single folder: convert JSON files to inference_outputs format"""
    folder_name = os.path.basename(folder_path)
    print(f"\nProcessing folder: {folder_name}")
    
    # Look for the main JSON file (e.g., arc_easy_qwen.json)
    json_pattern = f"{folder_name}_qwen.json"
    input_json_path = os.path.join(folder_path, json_pattern)
    
    if not os.path.exists(input_json_path):
        print(f"  JSON file not found: {json_pattern}, skipping...")
        return
    
    # Replace the original file
    output_json_path = input_json_path
    
    # Convert the JSON format
    success = convert_json_format(input_json_path, output_json_path)
    
    if success:
        print(f"  ✅ Successfully processed {folder_name}")
    else:
        print(f"  ❌ Failed to process {folder_name}")

def main():
    """Main function to process all folders"""
    # Use relative path to work in current workspace
    base_path = "data/qwen2_narrow_cost_spread"
    
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
    processed_count = 0
    for folder in sorted(folders):
        folder_path = os.path.join(base_path, folder)
        try:
            process_folder(folder_path)
            processed_count += 1
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
    print(f"  Processed folders: {processed_count}/{len(folders)}")
    print(f"  JSON files converted: {len(json_files)}")
    
    if json_files:
        print(f"\nConverted JSON files:")
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    print(f"  - {os.path.basename(json_file)}: {len(data)} entries")
            except:
                print(f"  - {os.path.basename(json_file)}: Error reading file")

if __name__ == "__main__":
    main()