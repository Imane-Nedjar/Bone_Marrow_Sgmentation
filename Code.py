#Final test+graphe
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import distance_transform_edt as distance
from scipy.spatial.distance import directed_hausdorff
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION ---
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 100
IMG_PATH = "/kaggle/input/datasets/alzaouti/isnbdataset/ISNB/images"
MASK_PATH = "/kaggle/input/datasets/alzaouti/isnbdataset/ISNB/mask"

def clear_gpu():
    torch.cuda.empty_cache()
    gc.collect()

# --- 2. ARCHITECTURE: BOTTLENECK ATTENTION ---
class BottleneckAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(BottleneckAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- 3. STABLE HYBRID LOSS ---
class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()
    def forward(self, probs, dist_maps):
        pc = probs[:, 1:, :, :] 
        dc = dist_maps[:, 1:, :, :]
        return torch.mean(pc * dc) 

def compute_dist_map_batch(mask_batch):
    batch_size, h, w = mask_batch.shape
    dist_maps = np.zeros((batch_size, 3, h, w)) 
    for b in range(batch_size):
        for c in range(3):
            posmask = (mask_batch[b] == c).cpu().numpy().astype(np.uint8)
            if posmask.any():
                negmask = 1 - posmask
                res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
                dist_maps[b, c] = np.tanh(res / 5.0) 
    return torch.from_numpy(dist_maps).float().to(DEVICE)

# --- 4. DATASET ---
def rgb_to_mask(rgb_mask):
    mask = np.zeros((rgb_mask.shape[0], rgb_mask.shape[1]), dtype=np.uint8)
    yellow = (rgb_mask[:,:,0] > 150) & (rgb_mask[:,:,1] > 150) & (rgb_mask[:,:,2] < 100)
    green = (rgb_mask[:,:,1] > 150) & (rgb_mask[:,:,0] < 100) & (rgb_mask[:,:,2] < 100)
    mask[yellow], mask[green] = 1, 2
    return mask

class BloodCellDataset(Dataset):
    def __init__(self, img_files, mask_files, transform=None):
        self.img_files, self.mask_files = img_files, mask_files
        self.transform = transform
    def __len__(self): return len(self.img_files)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.img_files[idx]), cv2.COLOR_BGR2RGB)
        mask = rgb_to_mask(cv2.cvtColor(cv2.imread(self.mask_files[idx]), cv2.COLOR_BGR2RGB))
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
        return img, mask.long()

# --- 5. INITIALIZATION ---
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=3)
model.encoder.layer4.add_module("Bottleneck_Attention", BottleneckAttention(512))
model = model.to(DEVICE)

dice_criterion = smp.losses.DiceLoss(mode='multiclass')
boundary_criterion = BoundaryLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# --- 6. ENGINES ---
history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': [], 
           'val_precision': [], 'val_recall': [], 'val_accuracy': []}

def train_model(train_loader, val_loader):
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            dist_maps = compute_dist_map_batch(masks)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(imgs)
                loss = dice_criterion(out, masks) + 1.0 * boundary_criterion(F.softmax(out, dim=1), dist_maps)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            train_loss += loss.item()

        model.eval()
        v_dice, v_iou, v_prec, v_rec, v_acc, v_loss = 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                out = model(imgs)
                tp, fp, fn, tn = smp.metrics.get_stats(out.argmax(1), masks, mode='multiclass', num_classes=3)
                v_dice += smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro").item()
                v_iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro").item()
                v_prec += smp.metrics.precision(tp, fp, fn, tn, reduction="macro").item()
                v_rec += smp.metrics.recall(tp, fp, fn, tn, reduction="macro").item()
                #v_acc += smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro").item()
                v_loss += dice_criterion(out, masks).item()

        n_val = len(val_loader)
        history['train_loss'].append(train_loss/len(train_loader))
        history['val_loss'].append(v_loss/n_val); history['val_dice'].append(v_dice/n_val)
        history['val_iou'].append(v_iou/n_val); history['val_precision'].append(v_prec/n_val)
        history['val_recall'].append(v_rec/n_val); history['val_accuracy'].append(v_acc/n_val)
        print(f"Epoch {epoch+1:02d} | T-Loss: {history['train_loss'][-1]:.4f} | Dice: {history['val_dice'][-1]:.4f}")
        clear_gpu()

def evaluate_test_set(test_loader):
    model.eval()
    t_dice, t_iou, t_prec, t_rec, t_acc = 0, 0, 0, 0, 0
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            out = model(imgs)
            tp, fp, fn, tn = smp.metrics.get_stats(out.argmax(1), masks, mode='multiclass', num_classes=3)
            t_dice += smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro").item()
            t_iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro").item()
            t_prec += smp.metrics.precision(tp, fp, fn, tn, reduction="macro").item()
            t_rec += smp.metrics.recall(tp, fp, fn, tn, reduction="macro").item()
            t_acc += smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro").item()
    n = len(test_loader)
    print("\n--- Final Test Evaluation ---")
    res = {"F1/Dice": t_dice/n, "IoU": t_iou/n, "Precision": t_prec/n, "Recall": t_rec/n, "Pixel Accuracy": t_acc/n}
    for m, s in res.items(): print(f"{m:<20} | {s:.4f}")

# --- 7. VISUALIZATION ---
def visualize_test_samples(test_ds, num_samples=10):
    model.eval()
    color_map = {0: [0,0,0], 1: [255,255,0], 2: [0,255,0]} 
    for i in range(min(num_samples, len(test_ds))):
        img_t, mask_t = test_ds[i]
        with torch.no_grad():
            pred = model(img_t.unsqueeze(0).to(DEVICE)).argmax(1).squeeze(0).cpu().numpy()
        
        hd_scores = []
        for c in [1, 2]:
            p, t = np.argwhere(pred == c), np.argwhere(mask_t.numpy() == c)
            if p.any() and t.any():
                hd_scores.append(max(directed_hausdorff(p, t)[0], directed_hausdorff(t, p)[0]))
        avg_hd = np.mean(hd_scores) if hd_scores else 0.0

        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        img_disp = (img_t.permute(1,2,0).numpy() * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0,1)
        gt_rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        pr_rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        for c in range(3):
            gt_rgb[mask_t.numpy() == c] = color_map[c]
            pr_rgb[pred == c] = color_map[c]

        ax[0].imshow(img_disp); ax[0].set_title(f"Test Image {i+1}"); ax[0].axis('off')
        ax[1].imshow(gt_rgb); ax[1].set_title("Ground Truth Mask"); ax[1].axis('off')
        ax[2].imshow(pr_rgb); ax[2].set_title(f"Prediction (Avg HD: {avg_hd:.2f})"); ax[2].axis('off')
        plt.savefig(f'test_sample_result_{i+1}.png', dpi=300, bbox_inches='tight')
        plt.show(); plt.close()

# --- 7. PLOTTING: JOURNAL QUALITY ---
def save_results_plots(history):
    plt.rcParams.update({'font.family': 'serif', 'font.size': 14})
    
    # 1. Loss Figure
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], 'r--', linewidth=2, label='Train Loss')
    plt.plot(history['val_loss'], 'b-', linewidth=2, label='Val Loss')
    plt.title('Loss Convergence Analysis'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig('Figure_Loss_Convergence.png', dpi=600, bbox_inches='tight')
    plt.show()

    # 2. Metrics Figure
    plt.figure(figsize=(10, 6))
    plt.plot(history['val_dice'], color='#2ca02c', marker='o', markersize=3, label='Dice Score')
    plt.plot(history['val_iou'], color='#9467bd', marker='s', markersize=3, label='IoU')
    #plt.plot(history['val_accuracy'], color='#d62728', linestyle='--', label='Pixel Accuracy') # Added to Plot
    plt.plot(history['val_precision'], color='#ff7f0e', label='Precision')
    plt.plot(history['val_recall'], color='#17becf', label='Recall')
    plt.title('Segmentation Performance Metrics'); plt.xlabel('Epochs'); plt.ylabel('Score'); plt.ylim(0, 1.05); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig('Figure_Segmentation_Accuracy.png', dpi=600, bbox_inches='tight')
    plt.show()

# --- 8. EXECUTION ---
imgs = sorted([os.path.join(IMG_PATH, f) for f in os.listdir(IMG_PATH) if f.endswith('.bmp')])
msks = sorted([os.path.join(MASK_PATH, f) for f in os.listdir(MASK_PATH) if f.endswith('.bmp')])

if imgs:
    tr_i, rem_i, tr_m, rem_m = train_test_split(imgs, msks, test_size=0.2, random_state=42)
    vl_i, ts_i, vl_m, ts_m = train_test_split(rem_i, rem_m, test_size=0.5, random_state=42)
    
    train_loader = DataLoader(BloodCellDataset(tr_i, tr_m, transform=transforms), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(BloodCellDataset(vl_i, vl_m, transform=transforms), batch_size=BATCH_SIZE)
    test_ds = BloodCellDataset(ts_i, ts_m, transform=transforms)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    train_model(train_loader, val_loader)
    evaluate_test_set(test_loader)
    save_results_plots(history)
    visualize_test_samples(test_ds) # Generating qualitative visual results