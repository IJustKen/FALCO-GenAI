import os
import json
import glob
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from lib import ArcFace

def calc_baselines():
    device = 'cuda'
    model = ArcFace().eval().to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(256)
    ])

    # Updated path for the aligned originals and reconstructions
    inv_dir = "datasets/inv/celebahq/data"

    # Auto-detect the fake directory and map
    fake_dirs = glob.glob("datasets/fake/*")
    fake_dir = fake_dirs[0] if fake_dirs else ""
    map_files = glob.glob(os.path.join(fake_dir, "*celebahq.json"))
    nn_map_file = map_files[0] if map_files else ""

    with open(nn_map_file, 'r') as f:
        nn_map = json.load(f)

    fake_cosines = []
    inv_cosines = []

    for orig_name, fake_name in nn_map.items():
        # Extract the base number (e4e strip logic)
        name_base = os.path.splitext(orig_name)[0]
        
        # Point directly to the inv folder's aligned and recon images
        orig_path = os.path.join(inv_dir, f"{name_base}_aligned.jpg")
        inv_path = os.path.join(inv_dir, f"{name_base}_recon.jpg")
        
        # Fake NN path (based on lib/celebahq.py logic)
        fake_nn_path = os.path.join(fake_dir, fake_name, 'image.jpg')
        
        if not os.path.exists(orig_path): 
            continue
        
        orig_img = transform(Image.open(orig_path).convert('RGB')).unsqueeze(0).to(device)
        
        with torch.no_grad():
            orig_feat = model(orig_img)
            
            # Baseline A: Fake NN
            if os.path.exists(fake_nn_path):
                print(fake_nn_path)
                fake_img = transform(Image.open(fake_nn_path).convert('RGB')).unsqueeze(0).to(device)
                fake_feat = model(fake_img)
                fake_cosines.append(F.cosine_similarity(orig_feat, fake_feat).item())
                
            # Baseline B: Plain e4e Inversion
            if os.path.exists(inv_path):
                print(inv_path)
                inv_img = transform(Image.open(inv_path).convert('RGB')).unsqueeze(0).to(device)
                inv_feat = model(inv_img)
                inv_cosines.append(F.cosine_similarity(orig_feat, inv_feat).item())

    if fake_cosines:
        print(f"Baseline (a) Fake NN -> Mean ArcFace Cosine: {sum(fake_cosines)/len(fake_cosines):.4f}")
    if inv_cosines:
        print(f"Baseline (b) e4e Inversion -> Mean ArcFace Cosine: {sum(inv_cosines)/len(inv_cosines):.4f}")