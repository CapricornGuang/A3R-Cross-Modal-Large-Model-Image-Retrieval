import csv, os


def parse_aug_auto(aug_file_path, save_file_path, separator, root_path):
    '''
    transform the augmentation file to the required format of CLIP model
    '''
    with open(aug_file_path, "r") as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]
    with open(save_file_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["filepath", "title"])
        for line in lines:
            parts = line.strip().split(separator)
            filepath = parts[0]
            caption = parts[-1]
            writer.writerow([os.path.join(root_path, filepath), caption])
    print(f'{aug_file_path} is sucessfully parserd')
    print(f'parsed contents are saved to {save_file_path}')
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--aug_file_path', type=str, required=True, 
                    help='the path of the augmentation file')
    parser.add_argument('--save_file_path', type=str, required=True, 
                    help='the path to save the parsing file of the augmentation file')
    
    parser.add_argument('--separator', type=str, required=True, 
                    help='separator of the input file')
    
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'val'], 
                    help='separator of the input file')
    
    args = parser.parse_args()
    root_path = f'data/datasets/{args.mode}/{args.mode}_images/'
    parse_aug_auto(args.aug_file_path, args.save_file_path, args.separator, root_path)
