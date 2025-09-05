# infer_and_plot.py
import os, re, glob, argparse, numpy as np, torch
from pathlib import Path
import matplotlib.pyplot as plt

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, EnsureTyped, Activations, AsDiscrete
)
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
import torch.nn.functional as F


#model config match the training
UNET_CONFIG = dict(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
)

def load_state_dict_smart(net, ckpt_path, device):
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict):
        if "model_state_dict" in obj:
            net.load_state_dict(obj["model_state_dict"])
        else:
            net.load_state_dict(obj)
    else:
        net.load_state_dict(obj)
    #print model info
    n_params = sum(p.numel() for p in net.parameters())
    print(f"? Loaded checkpoint: {ckpt_path} | #Params: {n_params}")
    return net


def get_case_files(data_dir, case_id):
    """
    Return dict {"image": <path>, "label": <path>} for one case.
    Handles both exact names and zero-padded variants via glob fallback.
    """
    root = Path(data_dir)
    img = root / "imagesTr" / f"spleen_{int(case_id)}.nii.gz"
    lab = root / "labelsTr" / f"spleen_{int(case_id)}.nii.gz"

    if not img.exists() or not lab.exists():
        # fallback: handle zero-padding / different extensions
        imgs = sorted((root / "imagesTr").glob(f"spleen_{case_id}*.nii*"))
        labs = sorted((root / "labelsTr").glob(f"spleen_{case_id}*.nii*"))
        assert imgs and labs, f"Could not find files for case {case_id} under {root}"
        img, lab = imgs[0], labs[0]

    return {"image": str(img), "label": str(lab)}

    

def build_val_tranform(args):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=map(float, args.spacing.split(",")),
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=["image", "label"]),
    ])
    
def choose_slice_with_fg(mask_3d):
    # mask_3d: [D, H, W] {0, 1}
    sums = mask_3d.reshape(mask_3d.shape[0], -1).sum(axis=1)
    return int(np.argmax(sums)) if sums.max() > 0 else mask_3d.shape[0] // 2
        
def dice_binary(pred, gt):
    # pred, label: [D, H, W] {0, 1}
    pred, gt = pred.astype(bool), gt.astype(bool)
    intersect = np.logical_and(pred, gt).sum()
    denom = np.logical_or(pred, gt).sum()
    return 2.0 * intersect / (denom + 1e-8)

def plot_triplet(image3d, gt3d, pred3d, slice_idx, save_path=None, title_prefix=""):
    img = image3d[slice_idx]
    gt = gt3d[slice_idx]
    pred = pred3d[slice_idx]
    
    plt.figure(figsize=(12, 4))
    #Image
    ax = plt.subplot(1, 3, 1)
    ax.imshow(img, cmap="gray")
    # ax.set_title(f"{title_prefix}Image (z={slice_idx})"); ax.axis("off")
    #Ground truth overlay
    ax = plt.subplot(1, 3, 2)
    ax.imshow(img, cmap="gray")
    ax.imshow(np.ma.masked_where(gt==0, gt), alpha=0.3, cmap="Purples")
    ax.set_title(f"{title_prefix}Ground Truth"); ax.axis("off")
    #prediction overlay
    ax = plt.subplot(1, 3, 3)
    ax.imshow(img, cmap="gray")
    ax.imshow(np.ma.masked_where(pred==0, pred), alpha=0.3, cmap="Purples")
    # ax.set_title(f"{title_prefix}Prediction"); ax.axis("off")
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"? Saved figure -> {save_path}")
    else:
        plt.show()
    plt.close()
    
def compute_metrics_3d(pred, gt, spacing=(1.5, 1.5, 1.5)):
    #pred, gt: [D, H, W] {0, 1}
    pred_t = torch.as_tensor(pred.astype(np.int64)) #[D, H, W]
    gt_t = torch.as_tensor(gt.astype(np.int64)) #[D, H, W]
    
    pred_t = pred_t.unsqueeze(0) #[1, D, H, W]
    gt_t = gt_t.unsqueeze(0) #[1, D, H, W]
    
    #on hot
    num_classes = 2
    pred_t = F.one_hot(pred_t, num_classes=num_classes) #[1, D, H, W, 2]
    pred_t = pred_t.permute(0, 4, 1, 2, 3).float() #[1, 2, D, H, W]

    gt_t = F.one_hot(gt_t, num_classes=num_classes).float() #[1, D, H, W, 2]
    gt_t = gt_t.permute(0, 4, 1, 2, 3).float() #[1, 2, D, H, W]
    
    #dice
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    print(f"? Computing Dice ...")
    dice_metric(pred_t, gt_t)
    dsc = dice_metric.aggregate().item()
    dice_metric.reset()
    print(f"  Dice: {dsc:.4f}")
    
    #hausdorff
    haus_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    print(f"? Computing Hausdorff95 ...")
    haus_metric(pred_t, gt_t, spacing=spacing)
    hd95 = haus_metric.aggregate().item()
    haus_metric.reset()
    print(f"  Hausdorff95: {hd95:.4f} mm")
    
    #assd
    assd_metric = SurfaceDistanceMetric(include_background=False, reduction="mean")
    print(f"? Computing ASSD ...")
    assd_metric(pred_t, gt_t, spacing=spacing)
    assd = assd_metric.aggregate().item()
    assd_metric.reset()
    print(f"  ASSD: {assd:.4f} mm")
    
    return {"dice": dsc, "hausdorff95": hd95, "assd": assd}
    
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="./Task09_Spleen",
                    help="data root that contains Task09_Spleen folder")
    ap.add_argument("--ckpt", type=str, default="./runs/spleen_unet/best_weights.pt",
                    help="model checkpoint path")
    ap.add_argument("--case-idx", type=int, default=10,
                    help="which case to run inference on")
    ap.add_argument("--roi", default="128, 128, 128", 
                    help="sliding window roi size, as D,H,W")
    ap.add_argument("--sw-batch", type=int, default=4,
                    help="sliding window batch size")
    ap.add_argument("--save", default="./runs/spleen_unet/preview.png",
                    help="where to save the inference result figure")
    ap.add_argument("--spacing", default="1.5,1.5,1.5",
                    help="resample spacing for val transform, as x,y,z")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.cuda.is_available()
    
    #build model and load checkpoint
    net = UNet(**UNET_CONFIG).to(device)
    
    if not os.path.exists(args.ckpt):
        fallback = Path(args.ckpt).parent / "best.pt"
        assert fallback.exists(), f"ckpt not found: {args.ckpt}"
        print(f"  ! Using fallback ckpt: {fallback}")
        load_state_dict_smart(net, fallback, device)
    else:
        load_state_dict_smart(net, args.ckpt, device)
        
    #build validation data
    case = get_case_files(args.data_dir, args.case_idx)
    #check case_idx is the same as spleen_caseid.nii.gz
    assert f"spleen_{int(args.case_idx)}" in Path(case["image"]).name, \
        f"case_idx {args.case_idx} does not match image file {case['image']}"
        
    val_tf = build_val_tranform(args)
    #load single case
    sample = val_tf(case)
    
    img_t, lab_t = sample["image"].unsqueeze(0).to(device), \
                    sample["label"].unsqueeze(0).to(device)  #add batch dim #[1, 1, D, H, W]
    
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=amp):
        D, H, W = img_t.shape[2:]
        rd, rh, rw = map(int, args.roi.split(","))
        logit = sliding_window_inference(img_t, (rd, rh, rw), args.sw_batch, net) #[1, 2, D, H, W]
    #post process (softmax -> argmax)
    prob = torch.softmax(logit, dim=1) #[1, 2, D, H, W]
    pred = torch.argmax(prob, dim=1, keepdim=False) #[1, D, H, W]
    pred_np = pred[0].float().cpu().numpy() #[D, H, W]
    img_np = img_t[0, 0].float().cpu().numpy() #[D, H, W]
    gt_np = lab_t[0, 0].float().cpu().numpy() #[D, H, W]
    spacing = tuple(map(float, args.spacing.split(",")))
    metrics = compute_metrics_3d(pred_np, gt_np, spacing=spacing)
    

    z = choose_slice_with_fg(gt_np)
    title = f"Case {args.case_idx} | Dice: {metrics['dice']:.4f} | HD95mm: {metrics['hausdorff95']:.2f}mm | ASSDmm: {metrics['assd']:.2f}mm\n"
    plot_triplet(img_np, gt_np, pred_np, z, save_path=args.save, title_prefix=title)
        
if __name__ == "__main__":
    main()
