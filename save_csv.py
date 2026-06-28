import os
import glob
import re
import pandas as pd

# Define the exact list of columns you want to apply
NEW_COLUMNS = [
    'x', 'y', 'z', 'omnivariance', 'eigenentropy', 'anisotropy', 
    'planarity', 'linearity', 'surface_variation', 'sphericity', 
    'verticality', 'epistemic', 'label'
]

def process_space_separated_csvs(directory_path):
    """
    Opens single-column space-separated text files, splits them into 13 structured columns,
    normalizes x/y coordinates to a local origin of 0, and saves as standard CSVs.
    """
    search_path = os.path.join(directory_path, "*.csv")
    csv_files = sorted(glob.glob(search_path))
    
    if not csv_files:
        print(f"No .csv files found in directory: {directory_path}")
        return

    print(f"Found {len(csv_files)} files to process.\n")

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        try:
            # 1. Read the file as a single text column to avoid pandas misinterpreting broken lines
            # 'sep' is set to a dummy delimiter to pull the entire row into a single string series
            df_raw = pd.read_csv(file_path, sep="§", header=None, engine='python')
            
            # Extract the raw text rows
            raw_lines = df_raw.iloc[:, 0].astype(str)
            
            # 2. Parse lines using regular expressions to handle variable whitespace/multiple spaces
            parsed_data = []
            for line in raw_lines:
                clean_line = line.strip()
                if not clean_line:
                    continue
                # Split by one or more whitespace characters
                split_row = re.split(r'\s+', clean_line)
                
                # Check for row structural integrity
                if len(split_row) == len(NEW_COLUMNS):
                    parsed_data.append(split_row)
            
            # If the original file had an alphanumeric text header row (like 'x y z...'),
            # the conversion to float below will fail. Let's discard it if found:
            try:
                float(parsed_data[0][0])
            except ValueError:
                # The first row is text headers; pop it out
                parsed_data.pop(0)

            # 3. Create a clean structured DataFrame
            df = pd.DataFrame(parsed_data, columns=NEW_COLUMNS, dtype=float)
            
            # 4. Apply min-subtraction normalization to shift the local tile origin to (0,0)
            df['x'] = df['x'] - df['x'].min()
            df['y'] = df['y'] - df['y'].min()
            
            # Ensure label is cast as integer types for the dataset builder
            df['label'] = df['label'].astype(int)
            
            # 5. Save the file back out as a standard comma-separated format
            df.to_csv(file_path, index=False, sep=",")
            print(f"  [Success] Parsed, normalized, and saved {filename} ({len(df)} rows)")
            
        except Exception as e:
            print(f"  [Error] Failed to process {filename}: {e}")

if __name__ == "__main__":
    # Update this path to match your data directory
    TARGET_DIR = r"F:\Aditya\Tiles\Toronto Tiles\split_tiles"
    
    process_space_separated_csvs(TARGET_DIR)