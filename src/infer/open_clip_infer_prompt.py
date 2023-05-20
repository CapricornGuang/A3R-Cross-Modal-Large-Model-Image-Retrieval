import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.data import Dataset, DataLoader

from PIL import Image
import urllib.request
import os, math, gc, sys, json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import argparse



parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, required=True, 
                    help='pretrained model name')
                    
parser.add_argument('--pt_path', type=str, required=True, 
                    help='pretrained model path')
                    
parser.add_argument('--image_root', type=str, required=True, 
                    help='test set image root')
                    
parser.add_argument('--text_path', type=str, required=True, 
                    help='test set text path')
                    
parser.add_argument('--run_name', type=str, required=True, choices=['open_clip_car_infer', 'open_clip_person_infer'], 
                    help='the output json file will be named with \{run_name\}.json')
                    
parser.add_argument('--text_steps', type=int, default=600, 
                    help='text steps when loading test set')
                    
parser.add_argument('--image_batch', type=int, default=400, 
                    help='image batch when loading test set')

parser.add_argument('--topk', type=int, default=10, 
                    help='get topK value for a query text')
                    
args = parser.parse_args()

# 导入CLIP
import open_clip

device = "cuda:0" if torch.cuda.is_available() else "cpu"


print('model loading...')
"""Open-CLIP --epoch 30 --batchsize 50"""

md_name = args.model_name
pt_path = args.pt_path
images_root = args.image_root
texts_root = args.text_path
run_name = args.run_name

image_batch = args.image_batch
text_steps = args.text_steps



def square_padding(image):
    w, h = image.size
    pad_w = max(0, (h - w) // 2)
    pad_h = max(0, (w - h) // 2)
    image = T.Pad((pad_w, pad_h))(image)
    return image
    
    
# model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained = pt_path, device=device)
# text_tokenizer = open_clip.get_tokenizer('ViT-L-14')
model, _, image_preprocess = open_clip.create_model_and_transforms(model_name=md_name, pretrained=pt_path, device=device)
image_preprocess = T.Compose([T.Lambda(square_padding)]+image_preprocess.transforms)
print(image_preprocess)

text_tokenizer = open_clip.get_tokenizer(md_name)

#@title ## CLIP-Image/Text Tokenizer
class imageTokenizer(nn.Module):
    def __init__(self, device):
        super(imageTokenizer, self).__init__()
        self.device = device

    @staticmethod
    def load_image(img_path, resize=None):
        image = image_preprocess(Image.open(img_path))
        # image = Image.open(img_path).convert("RGB")
        # if resize is not None:
        #     image = image.resize((resize, resize))
        # image_tensor = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.)
        # image_tensor = image_tensor.permute(2, 0, 1)
        return image

    
#@title ##构造Pytorch Dataset

def replace_prefix(sentence, prefixs):
    sentence_replaced = sentence
    for prefix in prefixs:
        if sentence.startswith(prefix):
            sentence_replaced = sentence.replace(prefix, "", 1)
    return sentence_replaced


class ImageRetrivalTestSet(Dataset):
    def __init__(self, images_root, texts_root, image_resize=None):
        #Load text data ONCE
        self.texts_root = texts_root
        self.texts = ImageRetrivalTestSet._load_texts(texts_root)#[:20]
        self.ori_texts = ImageRetrivalTestSet._load_original_texts(texts_root)
        #Load image path ONCE, the image data should be loaded IN BATCH
        self.images_root = images_root
        self.image_paths = os.listdir(images_root)#[:200]
        self.idx_dic = self._get_images_idx()
        self.image_resize = image_resize

    # @staticmethod
    # def _load_texts(texts_root):
    #     f = open(texts_root, 'r')
    #     texts = []
    #     for line in tqdm(f.readlines()):
    #         text = line.strip() #delet \n
    #         texts.append(text)
    #     return texts
    
    @staticmethod
    def _load_texts(texts_root):
        prefixs = ["A ", "This is a ", "An ", "This is an "]
        f = open(texts_root, 'r')
        texts = []
        for line in tqdm(f.readlines()):
            text = line.strip()[:-1] #delet \n and "."
            if run_name == 'open_clip_car_infer':
                text = text + " image taken by traffic surveillance cameras"   #replace_prefix(text, prefixs)
            texts.append(text)
        return texts


    @staticmethod
    def _load_original_texts(texts_root):
        f = open(texts_root, 'r')
        texts = []
        for line in tqdm(f.readlines()):
            text = line.strip()
            texts.append(text)
        return texts
        
    def _get_images_idx(self):
        absolut_image_paths = [image_name for image_name in self.image_paths]
        return {i: path for i, path in enumerate(absolut_image_paths)}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        # Load the image from disk
        image_name = self.image_paths[index]
        image_path = os.path.join(self.images_root, image_name)
        image_np = imageTokenizer.load_image(image_path, self.image_resize)
        return image_np
    
def get_images_texts_similarities(dataset, model, device, image_batch, text_steps):
    torch.cuda.empty_cache()
    # Create dataloader for images
    dataloader = DataLoader(dataset, batch_size=image_batch, shuffle=False)

    # Get the number of texts
    texts = dataset.texts
    num_texts = len(texts)
    print('query text number:', num_texts)
    print('key image number:', len(dataset))
    # Calculate the number of text batches based on batch size
    num_text_batches = math.ceil(num_texts / text_steps)

    # Initialize variable to store similarity matrix
    text_features = []
    #image_features = []
    sim_matrix_batch = []
    
    # Loop through text batches
    print('start loading text features')
    
    with torch.no_grad():
        for i in tqdm(range(num_text_batches)):
            # Get the start and end indices for the current text batch
            start_idx = i * text_steps
            end_idx = min(start_idx + text_steps, num_texts)

            # Tokenize and encode the current text batch
            # text_input = clip.tokenize(texts[start_idx:end_idx]).to(device)
            text_input = text_tokenizer(texts[start_idx:end_idx]).to(device)
            text_embed = model.encode_text(text_input).float() #batch_size, dim

            text_features.append(text_embed.detach().cpu())
    torch.cuda.empty_cache()
    text_features = torch.cat(text_features, dim=0)
    
    
    print('start loading image features')
    with torch.no_grad():
        for images_np in tqdm(dataloader):
            # Encode the current image batch
            images_np = images_np.to(device)
            image_embeds = model.encode_image(images_np)
            image_embeds = image_embeds.detach().cpu()
            sim_matrix_batch.append(torch.cosine_similarity(text_features.unsqueeze(1), image_embeds, dim=-1))
#             break
            
    torch.cuda.empty_cache()
    sim_matrix = torch.cat(sim_matrix_batch, dim=1)
    return sim_matrix


print('data loading...')
#@markdown V100-300,300; A100-600, 600
dataset = ImageRetrivalTestSet(images_root, texts_root)
sim_matrix = get_images_texts_similarities(dataset, model, device, 
          image_batch=image_batch, text_steps=text_steps)

def infer(similarity, idx_dic, texts):
    similarity_argsort = np.argsort(-similarity, axis=1)
    print('similarity_argsort.shape', similarity_argsort.shape)

    topk = args.topk
    result_list = []
    for i in range (len(similarity_argsort)):
        dic = {'text': texts[i], 'image_names': []}
        for j in range(topk):
            dic['image_names'].append(idx_dic[similarity_argsort[i,j]])
        result_list.append(dic)
    with open(f"{run_name}.json", 'w') as f:
        f.write(json.dumps({'results': result_list}, indent=4))
        
        
sim_matrix_np = sim_matrix.numpy()
texts = dataset.texts
ori_texts = dataset.ori_texts
idx_dic = dataset.idx_dic
infer(sim_matrix_np, idx_dic, ori_texts)