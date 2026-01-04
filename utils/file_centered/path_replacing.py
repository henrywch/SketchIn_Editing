"""
Path replacer as the datasets are moved once
"""

import json
import os
import argparse

# ================= CONFIGURATION =================
OLD_PATH = "/inspire/hdd/project/video-understanding/public/share"
NEW_PATH = "/inspire/hdd/project/video-understanding/public/personal/chwang"
# =================================================

def recursive_replace(obj, old_str, new_str):
    """
    Recursively traverse dictionaries and lists to replace strings.
    """
    if isinstance(obj, str):
        return obj.replace(old_str, new_str)
    
    elif isinstance(obj, dict):
        # Rebuild the dictionary with modified values
        # Note: If you also need to replace keys, change 'k' to k.replace(...)
        return {k: recursive_replace(v, old_str, new_str) for k, v in obj.items()}
    
    elif isinstance(obj, list):
        # Rebuild the list with modified items
        return [recursive_replace(i, old_str, new_str) for i in obj]
    
    else:
        # Return integers, floats, booleans, None as is
        return obj

def process_jsonl(input_file, output_file):
    print(f"Processing: {input_file}")
    print(f"Replacing: '{OLD_PATH}'")
    print(f"With:      '{NEW_PATH}'")
    
    count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                if not line.strip():
                    continue
                
                try:
                    # 1. Load the line as a dict
                    data = json.loads(line)
                    
                    # 2. Perform replacement recursively
                    new_data = recursive_replace(data, OLD_PATH, NEW_PATH)
                    
                    # 3. Write back to file
                    outfile.write(json.dumps(new_data) + "\n")
                    count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {e}")

        print(f"Success! Processed {count} lines.")
        print(f"Saved to: {output_file}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # You can run this script directly or pass arguments
    parser = argparse.ArgumentParser(description="Replace paths in a JSONL file.")
    parser.add_argument("--input_file", help="Path to the input .jsonl file")
    parser.add_argument("--output_file", help="Path to the output .jsonl file (optional)", default=None)
    
    args = parser.parse_args()
    
    # If output file is not provided, append '_modified' to input filename
    if not args.output_file:
        base, ext = os.path.splitext(args.input_file)
        out_path = f"{base}_modified{ext}"
    else:
        out_path = args.output_file

    process_jsonl(args.input_file, out_path)