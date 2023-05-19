from torch.utils.data import Dataset
from PIL import Image
import csv, re, json


def split_data(input_file, person_file, car_file):
    with open(person_file, 'w', newline='') as f_person, open(car_file, 'w', newline='') as f_car, open(input_file, 'r') as f_input:
        reader = csv.reader(f_input, delimiter='$')
        person_writer = csv.writer(f_person, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        car_writer = csv.writer(f_car, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for row in reader:
            img_name = row[0]
            attributes = row[1].split(',')
            text_desc = row[2]
            
            if len(attributes) == 21:
                attributes = [','.join(attributes)]
                person_writer.writerow([img_name] + attributes + [text_desc])
            else:
                attributes = [attributes[0][:-1]]
                car_writer.writerow([img_name] + attributes + [text_desc])
    print('data successfully split')
    return None

def get_car_attributes(car_file, car_attr_file):
    car_colors = set() 
    car_types =set()
    car_brands = set()

    with open(car_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                match = re.findall(r"^([a-z]+(?: [a-z]+)?) ([A-Z][a-zA-Z-]+(?: [A-Z][a-zA-Z-]+)*)|([A-Z][a-zA-Z-]+(?: [A-Z][a-zA-Z-]+)*)$", row[1])
                if match:
                    match_lst = match[0]
                    if match_lst[0] == '' and match_lst[1] == '':
                        brand = match_lst[2]
                        if brand == 'Hongyan':
                            car_brands.add(brand)
                        elif len(brand.split(' ')) != 1:
                            car_brands.add(brand)
                        else:
                            car_types.add(brand)
                    if match_lst[2] == '':
                        color = match_lst[0]
                        brand = match_lst[1]
                        car_colors.add(color)
                        car_brands.add(brand)
    
    car_colors = sorted(list(car_colors))
    car_brands = sorted(list(car_brands))
    car_types = sorted(list(car_types))
    
    data = {
        "Colors": car_colors,
        "Types": car_types,
        "Brands": car_brands,
    }
    print(len(car_colors), len(car_types), len(car_brands))
    print("Colors:", car_colors)
    print("Brands:", car_brands)
    print("Types:", car_types)
    
    with open(f"{car_attr_file}", "w") as jsonfile:
        json.dump(data, jsonfile)
    
    print(f"car attribute files are saved to {car_attr_file}")

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_label_path', type=str, required=True, 
                    help='the path of the label file for training')
                    
    parser.add_argument('--train_car_label', type=str, required=True, 
                    help='the path to save the label file for training car')
                    
    parser.add_argument('--train_person_label', type=str, required=True, 
                    help='the path to save the label file for training person')
    
    parser.add_argument('--car_attribute_path', type=str, required=True, 
                    help='the path to save the attribute of car')
    
                    
    args = parser.parse_args()
    
    train_input_file = args.train_label_path #'data/datasets/train/train_label.txt'
    train_car_file = args.train_car_label #'data/datasets/train/train_car_label.txt'
    train_person_file = args.train_person_label #'data/datasets/train/train_person_label.txt'
    car_attr_file = args.car_attribute_path #'data/car_attribute.json'
    
    split_data(train_input_file, train_person_file, train_car_file)
    get_car_attributes(train_car_file, car_attr_file)