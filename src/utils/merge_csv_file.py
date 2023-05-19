import pandas as pd

def merge_file(file1, file2, target_file, delimiter):
    
    df1 = pd.read_csv(file1, sep=delimiter)
    df2 = pd.read_csv(file2, sep=delimiter)

    df_merged = pd.concat([df1, df2])

    df_merged.to_csv(target_file, sep=delimiter, index=False)
    print(f"{file1} and {file2} are merged to {target_file}")
    
if __name__ == '__main__':
    import argparse
    
    import csv
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv_file1', type=str, required=True)
    parser.add_argument('--csv_file2', type=str, required=True)
    parser.add_argument('--csv_target', type=str, required=True)
    parser.add_argument('--delimiter', type=str, default= '\t')
    args = parser.parse_args()
    


    csv_file1 = args.csv_file1
    csv_file2 = args.csv_file2
    merged_csv_file = args.csv_target
    
    merge_file(csv_file1, csv_file2, merged_csv_file, args.delimiter)