import torch.nn as nn
from endterm.inference.flow_inference import FlowPriorSampler
from endterm.lib.latent_code import FlowLatentCode
from pathlib import Path

##### IMPORTS FROM ORIGINAL ####

import argparse
import torch
from torch.utils import data
from torch.optim.lr_scheduler import MultiStepLR
import os
import os.path as osp
from lib import DATASETS, CelebAHQ, DataParallelPassthrough, IDLoss, AttrLoss, LatentCode, tensor2image, anon_exp_dir
from models.load_generator import load_generator
from tqdm import tqdm
import json

import time
import csv

ROOT = Path(__file__).resolve().parent

# Helper to create the same output dir as the original code
def anon_exp_dir(args_dict, makedir=True):
    dir_name = osp.join('datasets', 'anon_flow', args_dict['dataset'],
                        '{}-{}_anon_{}_m-{}_lambda-id-{}_lambda-attr-{}_{}-{}_epochs-{}'.format(
                            args_dict['dataset'],args_dict['subset'],
                            args_dict['latent_space'],
                            args_dict['id_margin'],
                            args_dict['lambda_id'],
                            args_dict['lambda_attr'],
                            args_dict['optim'],
                            args_dict['lr'],
                            args_dict['epochs']))
    if makedir:
        os.makedirs(dir_name, exist_ok=True)

    return dir_name


def main():
    # copied this from original argparser
    parser = argparse.ArgumentParser(description="FALCO Anonymization with Flow-Matching Prior")
    parser.add_argument('-v', '--verbose', action='store_true', help="verbose mode on")
    parser.add_argument('--dataset', type=str, required=True, choices=DATASETS.keys(), help="choose real dataset")
    parser.add_argument('--dataset-root', type=str, help="set dataset root directory")
    parser.add_argument('--subset', type=str, default='train+val+test',
                        choices=('train', 'val', 'test', 'train+val', 'train+val+test'), help="choose dataset's subset")
    parser.add_argument('--fake-nn-map', type=str, required=False, help="fake NN map file (created by `pair_nn.py`)")
    parser.add_argument('--latent-space', type=str, default='W+', choices=('W+', 'S'), help="StyleGAN2's latent space")
    parser.add_argument('-m', '--id-margin', type=float, default=0.0, help="identity loss margin")
    parser.add_argument('--epochs', type=int, default=50, help="Number of anonymization steps")
    parser.add_argument('--optim', type=str, default='adam', choices=('sgd', 'adam'),
                        help="set optimizer ('sgd' or 'adam')")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--lr-milestones', nargs="+", default=[0.75, 0.9], help="learning rate scheduler milestones")
    parser.add_argument('--lr-gamma', type=float, default=0.8, help="learning rate decay parameter")
    parser.add_argument('--lambda-id', type=float, default=10.0, help="Scaling parameter of the ID loss")
    parser.add_argument('--lambda-attr', type=float, default=0.1, help="Scaling parameter of the attribute loss")
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.add_argument('--ckpt-path', default='NA', help="path to the model weights to be loaded for FM")
    parser.set_defaults(cuda=True)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    

    # Ensure output directories exist
    args_dict = args.__dict__.copy()
    out_dir = anon_exp_dir(args_dict)
    out_code_dir = osp.join(out_dir, 'latent_codes')
    out_data_dir = osp.join(out_dir, 'data')
    log_dir = ROOT / "endterm" / "logs" / f"m_{args.id_margin}"
    os.makedirs(out_code_dir, exist_ok=True)
    os.makedirs(out_data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create log file and write headers
    log_file = osp.join(log_dir, "convergence_log_flow.csv")
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['img_id', 'epoch', 'arcface_cos', 'id_loss', 'att_loss', 'wall_clock_time'])

    # Define which epochs to track
    # Note: args.epochs is usually 50, so the last epoch is 49.
    track_epochs = {0, 5, 10, 25, args.epochs - 1}

    print("Loading networks...")
    # Load StyleGAN generator
    G = load_generator(model_name='stylegan2_ffhq1024', latent_is_w=True, verbose=args.verbose).eval().to(device)
    
    # Load Loss Functions
    id_criterion = IDLoss(id_margin=args.id_margin).eval().to(device)
    attribute_loss = AttrLoss(feat_ext='farl', use_cuda=True).eval().to(device)

    # Load Flow Matching Sampler
    print("Loading Flow-Matching Prior...")
    flow_sampler = FlowPriorSampler(checkpoint_path=args.ckpt_path, device=device)

    # Load Dataloader
    if args.dataset == 'celebahq':
        dataset = CelebAHQ(root_dir=args.dataset_root,
                           subset=args.subset,
                           fake_nn_map=args.fake_nn_map,
                           inv=True)
        dataloader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False) 

    # ==========================================
    # THE OPTIMIZATION LOOP
    # ==========================================
    for data_idx, data_ in enumerate(tqdm(dataloader, desc="#. Anonymize {} images".format(args.dataset) if args.verbose else '')):

        # Get data, all code copied from original anonymize.py
        # but no fake NN this time
        img_orig = data_[0].to(device)
        img_orig_id = int(osp.basename(data_[2][0]).split('.')[0])
        img_recon_code = data_[6].to(device) # The 18-layer e4e real latent
        

        with torch.no_grad():
            # Extract the FaRL condition 'c' directly from the real image
            img_transformed = attribute_loss.feat_ext_transform(img_orig)
            c_real, _ = attribute_loss.extract_visual(img_transformed)

            # Sample the initialization from the Flow model (Returns shape: B, 5, 512)
            w_flow_init = flow_sampler.sample(condition_c=c_real, num_steps=20)

        # Build Flow anonymization latent code
        latent_code = FlowLatentCode(
            latent_code_real=img_recon_code, 
            latent_code_flow=w_flow_init, 
            img_id=img_orig_id,
            out_code_dir=out_code_dir, 
            latent_space='W+')
        
        latent_code.to(device)

        # Build optimizer
        optimizer = None
        if args.optim == 'sgd':
            optimizer = torch.optim.SGD(params=latent_code.parameters(), lr=args.lr)
        elif args.optim == 'adam':
            optimizer = torch.optim.Adam(params=latent_code.parameters(), lr=args.lr)

        # Set learning rate scheduler
        lr_scheduler = MultiStepLR(optimizer=optimizer,
                                   milestones=[int(m * args.epochs) for m in args.lr_milestones],
                                   gamma=args.lr_gamma)

        # Zero out gradients
        G.zero_grad()
        id_criterion.zero_grad()
        attribute_loss.zero_grad()

        start_time = time.time()    # start time for this image
        # Training (anonymization) loop for the current batch of images / latent codes
        for epoch in range(args.epochs):
            # Clear gradients wrt parameters
            optimizer.zero_grad()

            # Generate anonymized image
            img_anon = G(latent_code())

            # Calculate identity and attribute preservation losses
            id_loss = id_criterion(img_anon, img_orig)
            att_loss = attribute_loss(img_orig, img_anon)

            # Calculate total loss
            loss = args.lambda_id * id_loss + args.lambda_attr * att_loss

            # Convergence Logging
            if epoch in track_epochs:
                current_time = time.time() - start_time
                
                # Calculate the ArcFace cosine similarity
                with torch.no_grad():
                    # Use their exact transforms and feature extractor
                    feat_orig = id_criterion.extract_feats(id_criterion.id_transform(img_orig))
                    feat_anon = id_criterion.extract_feats(id_criterion.id_transform(img_anon))
                    
                    # Compute cosine similarity safely 
                    import torch.nn.functional as F
                    arcface_cos = F.cosine_similarity(feat_anon, feat_orig).item()
                
                with open(log_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        img_orig_id, 
                        epoch, 
                        f"{arcface_cos:.4f}", 
                        f"{id_loss.item():.4f}", 
                        f"{att_loss.item():.4f}", 
                        f"{current_time:.4f}"
                    ])

            # Back-propagation
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        # Store optimized anonymization latent codes
        latent_code.save()

        # Generate and save anonymized image
        with torch.no_grad():
            anonymized_image = G(latent_code())
            tensor2image(anonymized_image.cpu(), adaptive=True).save(
                osp.join(out_data_dir, '{}.jpg'.format(img_orig_id)),
                "JPEG", quality=75, subsampling=0, progressive=True
            )

if __name__ == '__main__':
    main()