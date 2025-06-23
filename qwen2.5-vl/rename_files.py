import os
import pandas as pd
from tqdm import tqdm

def main():
    # Specify the columns that contain the file paths
    columns = ['Lavie', 'Cogvid']
    df = pd.read_csv('./sorabench_data_with_local_paths.csv')

    for column in columns:
        paths = list(df[column])
        renamed_filenames = []  # This list will store the new filenames
        
        for i in tqdm(range(len(paths)), desc=f"Renaming files in column {column}"):
            path = paths[i]
            if isinstance(path, str) and len(path) > 0:
                # Create new file name based on the index
                new_filename = f"{i}.mp4"
                # Use the same directory as the original file, if any
                directory = os.path.dirname(path)
                new_path = os.path.join(directory, new_filename) if directory else new_filename

                # Try renaming the file and catch any potential errors
                try:
                    os.rename(path, new_path)
                    print(f"Renamed {path} to {new_path}")
                except Exception as e:
                    print(f"Error renaming {path} to {new_path}: {e}")
                    new_filename = ""  # Optionally, handle error by leaving an empty name
                renamed_filenames.append(new_filename)
            else:
                renamed_filenames.append('')
        
        # Add a new column to the dataframe for the new filenames
        df[f"{column}_renamed"] = renamed_filenames

    # Save the updated dataframe with the new renamed columns to a new CSV file
    output_csv = './sorabench_renamed.csv'
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV file saved to {output_csv}")

if __name__ == "__main__":
    main()
