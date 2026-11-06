import os
import glob
import cv2
import numpy as np
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import mediapipe as mp


def calc_leakage():
    # Initialize MediaPipe for quick face detection/masking
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # orig_dir = "datasets/inv/celebahq/data"
    # anon_dirs = glob.glob("datasets/anon/celebahq/*")
    orig_dir = "/home/s142301038-unnikrishnan/tard/FALCO/datasets/inv/celebahq/data"
    anon_dirs = glob.glob("/home/s142301038-unnikrishnan/tard/FALCO/datasets/anon_flow/celebahq/*")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # Standard StyleGAN [-1, 1] normalization
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(256)
    ])


    def get_masked_background(img_path):
        """Loads image, detects face, blacks it out, returns raw numpy array."""
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect face to create mask
        results = face_detector.process(img)
        mask = np.ones(img.shape[:2], dtype=np.uint8)
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                bw, bh = int(bbox.width * w), int(bbox.height * h)
                
                # Add a slight padding to the bounding box to ensure the whole head is masked
                pad = int(bw * 0.2)
                x1, y1 = max(0, x - pad), max(0, y - pad * 2)
                x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
                
                # Black out the face region
                mask[y1:y2, x1:x2] = 0
                
        # Apply mask
        bg_img = img * mask[:, :, np.newaxis]
        return bg_img

    for adir in sorted(anon_dirs):
        if not os.path.isdir(adir): continue
        
        m_val = adir.split('_m-')[1].split('_')[0] if '_m-' in adir else 'Unknown'
        ssim_scores = []
        mse_scores = []

        adir = os.path.join(adir, "data")
        
        for img_name in os.listdir(adir):
            #print(img_name)
            if not img_name.endswith('.jpg'): continue
            
            name_base = os.path.splitext(img_name)[0]
            orig_path = os.path.join(orig_dir, f"{name_base}_aligned.jpg")
            anon_path = os.path.join(adir, img_name)
            
            if not os.path.exists(orig_path): continue
            
            # Get masked backgrounds
            orig_bg_np = transform(get_masked_background(orig_path)).numpy()
            anon_bg_np = transform(get_masked_background(anon_path)).numpy()

            # print(orig_bg_np.shape)
            # print(anon_bg_np.shape)
            
            # Calculate SSIM (higher is more identical)
            # score_ssim, _ = ssim(orig_bg_np, anon_bg_np, channel_axis=2, full=True)
            # ssim_scores.append(score_ssim)
            
            # Calculate MSE (lower is more identical, 0 means exact match)
            score_mse = np.mean((orig_bg_np.astype("float") - anon_bg_np.astype("float")) ** 2)
            mse_scores.append(score_mse)
            
        #if ssim_scores:
            #mean_ssim = sum(ssim_scores)/len(ssim_scores)
        mean_mse = sum(mse_scores)/len(mse_scores)
        print(f"Margin m={m_val} -> BG MSE: {mean_mse:.4f} (0.0=Identical)")
            #print(f"Margin m={m_val} -> BG SSIM: {mean_ssim:.4f} (1.0=Identical) | BG MSE: {mean_mse:.4f} (0.0=Identical)")