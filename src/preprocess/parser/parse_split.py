import csv, os

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--txt_path', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    
    args = parser.parse_args()
    root_path = f'data/datasets/{args.mode}/{args.mode}_images/'
    
    with open(args.txt_path, "r") as txt_file:
        with open(args.csv_path, "w") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter="\t")
            csv_writer.writerow(["filepath", "title"])
            for line in txt_file:
                path, attr, caption = line.strip().split("$")
                csv_writer.writerow([os.path.join(root_path, path), caption])
                
                
    print(f"{args.txt_path} is successfully parsed")
    print(f"parsed contents are saved to {args.csv_path}" )