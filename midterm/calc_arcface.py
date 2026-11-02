import os
import glob
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from lib import ArcFace
import json

def calc_arcface():
    device = 'cuda'
    # Initialize the repo's exact ArcFace model
    model = ArcFace().eval().to(device)

    # Standard CelebA-HQ transforms used by the repo

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # Standard StyleGAN [-1, 1] normalization
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(256)
    ])

    # orig_dir = "datasets/CelebA-HQ/data"
    orig_dir = "/home/s142301038-unnikrishnan/tard/FALCO/datasets/CelebA-HQ/data" 
    anon_dirs = glob.glob("/home/s142301038-unnikrishnan/tard/FALCO/datasets/anon/celebahq/*")

    results = []

    for adir in sorted(anon_dirs):
        if not os.path.isdir(adir): continue
        
        # Extract the margin value from the directory name for printing
        m_val = adir.split('_m-')[1].split('_')[0] if '_m-' in adir else 'Unknown'
        cosines = []

        adir = os.path.join(adir, "data")

        img_wise = dict()

        for img_name in os.listdir(adir):

            if not img_name.endswith('.jpg'): continue
            
            orig_path = os.path.join(orig_dir, img_name)
            anon_path = os.path.join(adir, img_name)
            
            orig_img = transform(Image.open(orig_path).convert('RGB')).unsqueeze(0).to(device)
            anon_img = transform(Image.open(anon_path).convert('RGB')).unsqueeze(0).to(device)
            
            with torch.no_grad():
                orig_feat = model(orig_img)
                anon_feat = model(anon_img)
                
            cos_sim = F.cosine_similarity(orig_feat, anon_feat).item()
            cosines.append(cos_sim)
            img_wise[img_name] = cos_sim
        
        results.append(img_wise)
        # if cosines:
        print(f"Margin m={m_val} -> Mean ArcFace Cosine: {sum(cosines)/len(cosines):.4f}")

        #print(results)

    with open("datasets/anon/results.json", "w") as file:
        json.dump(results, file)   