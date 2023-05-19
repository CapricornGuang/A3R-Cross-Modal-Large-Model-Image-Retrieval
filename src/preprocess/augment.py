import os
import numpy as np
import argparse
from tqdm import tqdm
import itertools
import torch
from PIL import Image
import open_clip
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('data/car_attribute.json', 'r') as f:
    car_attr = json.load(f)
# car_attr = {"Colors": ["black", "blue", "brown", "green", "grey", "orange", "pink", "purple", "red", "white", "yellow"], "Types": ["Bus", "Microbus", "Minivan", "SUV", "Sedan", "Truck"], "Brands": ["Audi", "BAOJUN", "BESTUNE", "BMW", "BYD", "Balong Heavy Truck", "Bentley", "Benz", "Buick", "Cadillac", "Chana", "Chery", "Chevrolet", "China-Moto", "Citroen", "Dongfeng", "FAW", "FORLAND", "FOTON", "Ford", "Geely", "Golden Dragon", "GreatWall", "HAFEI", "Haima", "Honda", "Hongyan", "Hyundai", "Infiniti", "Isuzu", "Iveco", "JAC", "JMC", "Jeep", "Jinbei", "KINGLONG", "Karma", "Kia", "LEOPAARD", "Landrover", "Lexus", "Luxgen", "MORRIS-GARAGE", "Mazda", "Mini", "Mitsubishi", "Nissan", "OPEL", "PEUGEOT", "Porsche", "ROEWE", "SGMW", "SKODA", "Shacman", "Shuanghuan", "Soueast", "Style", "Subaru", "Suzuki", "Toyota", "Volkswagen", "Volvo", "XIALI", "Yutong", "ZXAUTO"]}
#print(car_attr)


def seek_one(text_list):
    for i in range(len(text_list)):
        if text_list[i] == 'a' or text_list[i] == 'A' or text_list[i] == 'an' or text_list[i] == 'An':
            return i
            
    return -1

def attr_traverse(split: str):
    text_list = []
    with open(os.path.join('data/datasets', split, split+'_label.txt')) as f:
        for text in f.readlines():
            text = text.replace('\n', '')
            text = text.replace('.', '')
            text_list.append(text)
        
    image_list = list(map(lambda x: x.split('$')[0], text_list))
    raw_attr_list = list(map(lambda x: x.split('$')[1], text_list))
    detail_list = list(map(lambda x: x.split('$')[2], text_list))
    
    
    color_list, brand_list, type_list = [], [], []
    for i in range(len(detail_list)):
        detail_text = detail_list[i]
        detail_text = list(detail_text.split(' '))
        if image_list[i][0] == '0':
            continue
            
        idx = seek_one(detail_text)
        attr_text = detail_text[idx+1:]
        
        tmp_color, tmp_brand, tmp_type = [], [], []
        for i in range(len(attr_text)):
            item = attr_text[i]
            
            if item == "Golden":
                tmp_brand.append("Golden Dragon")
                i = i + 1
                item = "Golden Dragonn"
            if item == "Balong":
                tmp_brand.append("Balong Heavy Truck")
                i = i + 2
                item = "Balong Heavy Truck"
            
            if item in car_attr["Colors"]:
                tmp_color.append(item)
            if item in car_attr["Brands"]:
                tmp_brand.append(item)
            if item in car_attr["Types"]:
                tmp_type.append(item)
                
        if tmp_color == [] and tmp_brand == []:
            tmp_color = car_attr["Colors"]
            #tmp_brand = car_attr["Brands"]
        elif tmp_color == [] and tmp_type == []:
            tmp_color = car_attr["Colors"]
            tmp_type = car_attr["Types"]
        elif tmp_brand == [] and tmp_type == []:
            tmp_brand = car_attr["Brands"]
            tmp_type = car_attr["Types"]
        elif tmp_color == []:
            tmp_color = car_attr["Colors"]
        elif tmp_brand == []:
            tmp_brand = car_attr["Brands"]
        elif tmp_type == []:
            tmp_type = car_attr["Types"]
            
        color_list.append(tmp_color)
        brand_list.append(tmp_brand)
        type_list.append(tmp_type)
            
    return image_list, raw_attr_list, detail_list, color_list, brand_list, type_list

   
def image_add_dot(image):
    '''
    asdasdjpg -> asdasd.jpg
    '''
    image = list(image)
    image.insert(-3, '.')
    return ''.join(image)
    

def attr_correction(args):
    bs = args.batchsize
    image_list, raw_attr_list, detail_list, color_list, brand_list, type_list = attr_traverse(args.split)
    image_list = [image_add_dot(image) for image in image_list]
    #print(color_list)
    assert len(color_list) % bs == 0
    
    f = open('data/augmented_'+ args.model + '_' + args.split + '_label.txt', 'a+')
    
    
    for idx in tqdm(range(0, len(color_list), bs)):
        num_batch = []
        text_list_batch = []
        image_batch = []
        model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
        tokenizer = open_clip.get_tokenizer(args.model)
        
        for i in range(idx, idx + bs):
            
        
            text_list = [d for d in itertools.product(color_list[i], brand_list[i], type_list[i])]
            
            if detail_list[i][0:4] == "This":
                text_list = ["This is a " + " ".join(text_list[j]) for j in range(len(text_list))]
            elif detail_list[i][0] == "A":
                text_list = ["A " + " ".join(text_list[j]) for j in range(len(text_list))]
            else:
                text_list = [" ".join(text_list[j]) for j in range(len(text_list))]
            
            image = preprocess(Image.open(os.path.join('data/datasets', args.split, args.split+'_images', image_list[i]))).unsqueeze(0)
            image = image.numpy().tolist()
            
            image_batch += image
            text_list_batch += text_list
            num_batch.append(len(text_list))
            

        
        model = model.to(device)
        image = torch.Tensor(image_batch).to(device)
        text = tokenizer(text_list_batch).to(device)
            
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            text_id = 0
            for i in range(bs):
                image_feature = image_features[i, :].unsqueeze(0)
                text_feature = text_features[text_id:text_id + num_batch[i], :]

                text_prob = torch.cosine_similarity(text_feature, image_feature.unsqueeze(1), dim=-1).softmax(dim=-1)
                text_prob = text_prob.to('cpu').numpy().tolist()
                
                j = np.array(text_prob[0]).argmax()
                print(f"Label probs: {text_prob}  argmax_id: {j}")
                final_attr = text_list_batch[text_id+j]
                f.write(f"{image_list[idx+i]}${raw_attr_list[idx+i]}${detail_list[idx+i]}.${final_attr}.\n")
                
                text_id += num_batch[i]
            
            
    f.close()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, required=True, default="2")
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'], default="train")
    parser.add_argument('--model', type=str, required=True, choices=['ViT-B-32', 'ViT-g-14', 'ViT-bigG-14'], default="ViT-bigG-14")
    parser.add_argument('--pretrained', type=str, required=True, choices=['laion2b_s34b_b79k', 'laion2b_s12b_b42k', 'laion2b_s39b_b160k'], default="laion2b_s39b_b160k")
    args = parser.parse_args()
    
    
    attr_correction(args)
