import os
import glob
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1

def calc_facenet():

    device = 'cuda'

    # The paper evaluates FaceNet pre-trained on VGGFace2 (and CASIA)
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # FaceNet expects 160x160 inputs and standard normalization
    transform = transforms.Compose([
        transforms.Resize((160, 160), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    orig_dir = "datasets/inv/celebahq/data"
    anon_dirs = glob.glob("datasets/anon/celebahq/*")

    for adir in sorted(anon_dirs):
        if not os.path.isdir(adir): continue
        
        m_val = adir.split('_m-')[1].split('_')[0] if '_m-' in adir else 'Unknown'
        cosines = []

        adir = os.path.join(adir, "data")
        
        for img_name in os.listdir(adir):
            if not img_name.endswith('.jpg'): continue
            
            # Match "5212.jpg" from anon folder to "5212_aligned.jpg" in inv folder
            name_base = os.path.splitext(img_name)[0]
            orig_path = os.path.join(orig_dir, f"{name_base}_aligned.jpg")
            anon_path = os.path.join(adir, img_name)
            
            if not os.path.exists(orig_path): continue
            
            orig_img = transform(Image.open(orig_path).convert('RGB')).unsqueeze(0).to(device)
            anon_img = transform(Image.open(anon_path).convert('RGB')).unsqueeze(0).to(device)
            
            with torch.no_grad():
                orig_feat = facenet(orig_img)
                anon_feat = facenet(anon_img)
                
            cos_sim = F.cosine_similarity(orig_feat, anon_feat).item()
            cosines.append(cos_sim)
            
        if cosines:
            print(f"Margin m={m_val} -> Mean FaceNet (VGGFace2) Cosine: {sum(cosines)/len(cosines):.4f}")