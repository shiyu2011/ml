#train_generalizeCT_unet.py
import os, glob, numpy as np, torch
import json, random
from pathlib import Path
import argparse


from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from monai.apps import download_and_extract
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld, 
    RandFlipd, RandRotate90d, EnsureTyped, Lambdad
)
from monai.utils import set_determinism
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric    
from monai.inferers import sliding_window_inference
from monai.networks.blocks import UpSample, UnetUpBlock

def get_files(data_dir, subFolder, run_dir, seed=0):
        #get all file names of images and labels
    imgs = sorted(glob.glob(os.path.join(data_dir, subFolder, "imagesTr", "*.nii.gz")))
    labs = sorted(glob.glob(os.path.join(data_dir, subFolder, "labelsTr", "*.nii.gz")))
    
    rng = np.random.RandomState(seed)
    idx = np.arange(len(imgs))
    rng.shuffle(idx)
    #20% for validation
    v = int(len(imgs) * 0.2)
    val_idx, train_idx = idx[:v], idx[v:]
    train_files = [{"image": imgs[i], "label": labs[i]} for i in train_idx]
    val_files = [{"image": imgs[i], "label": labs[i]} for i in val_idx]
    
    split = {
        "seed": seed,
        "train": [Path(imgs[i]).name for i in train_idx],
        "val": [Path(imgs[i]).name for i in val_idx]
    }
    
    with open(run_dir / "split.json", "w") as f:
        json.dump(split, f, indent=2)
    print(f"save split to {run_dir / 'split.json'}, train {len(train_files)}, val {len(val_files)}")
    return train_files, val_files


def save_ckpt(path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sceduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if amp else None,
        "best_dice": best_dice,
        "model_cfg": dict(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2
        ),
    }, path)


def print_param_list(net, idxSet=None):
    idx_map = []
    for i, (name, p) in enumerate(net.named_parameters()):
        if idxSet is None or i in idxSet:
            print(f"[{i:03d}] {name:80s} shape={tuple(p.shape)} requires_grad={p.requires_grad}")
            idx_map.append(name)
    return idx_map  # list of param names ordered by index



def set_param_requires_grad_by_index(net, idx, flag=True):
    for i, (name, p) in enumerate(net.named_parameters()):
        if i == idx:
            p.requires_grad = flag
            print(f"? set [{i:03d}] {name} requires_grad={flag}")
            return
            
def unfreeze_decoder_and_head(net, idx_set = []):
    #set all to False first
    for p in net.parameters():
        p.requires_grad = False
    print("Before unfreezing:")
    print_param_list(net)
    for idx in idx_set:
        set_param_requires_grad_by_index(net, idx, True)
    print("After unfreezing:")
    print_param_list(net)
    
            


def main():
    #arg
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", type=str, default="/home/rxm/ai-med-10day/", help="data root dir")
    ap.add_argument("--run-dir", type=str, default="./runs/liver_unet", help="where to save logs and models")
    ap.add_argument("--train-organ", type=str, default="liver", help="which organ to train")
    ap.add_argument("--is-fineTune", type=bool, default=True, help="whether to fine-tune from pretrained model")
    args = ap.parse_args()
    #setup
    seed = 0
    set_determinism(seed) # set nump, torch, random seeds
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.cuda.is_available()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    best_dice = float("-inf")
    
    def save_ckpt(path):
        torch.save({
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sceduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if amp else None,
            "best_dice": best_dice,
            "model_cfg": dict(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2
            ),
        }, path)
    
    # data prep
    root_dir = args.root_dir
    if args.train_organ.lower() == "spleen":
        url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
        subFolder = "Task09_Spleen"
    elif args.train_organ.lower() == "liver":
        url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar"
        subFolder = "Task03_Liver"
    else:
        raise ValueError(f"Unsupported organ {args.train_organ}, only spleen and liver are supported.")
        
    if not os.path.exists(os.path.join(root_dir, subFolder)):
        download_and_extract(url, os.path.join(root_dir, subFolder))
    
    train_files, val_files = get_files(root_dir, subFolder, run_dir, seed=0)
    
    #transforms
    train_tf = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=[1.5, 1.5, 1.5], mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=[0, 1, 2]),
        RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
        Lambdad(keys=["label"], func=lambda x: (x>0).astype(x.dtype)),
        EnsureTyped(keys=["image", "label"])
    ])
    
    val_tf = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=[1.5, 1.5, 1.5], mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True),
        Lambdad(keys=["label"], func=lambda x: (x>0).astype(x.dtype)),
        EnsureTyped(keys=["image", "label"])
    ])
    
    #datasets and loaders
    train_ds = CacheDataset(data=train_files, transform=train_tf, cache_rate=1.0, num_workers=4)
    val_ds = CacheDataset(data=val_files, transform=val_tf, cache_rate=1.0, num_workers=2)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    #model, loss, optim, metric
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    ).to(device)
    
    loss_fun = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=0.5, lambda_ce=0.5)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    sceduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", 
                                                          factor=0.5, 
                                                          patience=3,
                                                          threshold=1e-4,
                                                          threshold_mode="rel", 
                                                          min_lr=1e-6, 
                                                          verbose=True)
    
    scaler = GradScaler(enabled=amp)
    
    if not args.is_fineTune:
        resume_path = run_dir / "last.pt"
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device)
            net.load_state_dict(ckpt["model_state_dict"])
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            sceduler.load_state_dict(ckpt["scheduler_state_dict"])
            if amp and ckpt["scaler_state_dict"] is not None:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            best_dice = ckpt.get("best_dice", best_dice)
            print(f"Resumed from {resume_path}, epoch {ckpt['epoch']}, best_dice {best_dice:.4f}")
    else:    
    # if fine-tune, freeze encoder layers and early decoder but train the rest
        print(f"Fine-tuning from pretrained spleen model")
        spleen_pretrained_path = "./runs/spleen_unet/best_weights.pt"
        state_dict = torch.load(spleen_pretrained_path, map_location=device)
        net.load_state_dict(state_dict)
        unfreeze_decoder_and_head(net, idx_set = list(range(52, 63))) #unfreeze 
                         
        
    
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    # post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=2)])
    # post_label = AsDiscrete(to_onehot=2)
    
    roi_size = (128, 128, 128); sw_bs = 4
    epoches = 10
    
    for epoch in range(1, epoches + 1):
        net.train()
        running_loss = 0.0
        for b in train_loader:
            imgs, labels = b["image"].to(device), b["label"].to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                logit = net(imgs)
                loss = loss_fun(logit, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item()
            
        train_loss = running_loss / max(1, len(train_loader))
        
        
        #validation
        net.eval()
        dice_metric.reset()
        val_loss_sum = 0.0
        
        with torch.no_grad():
            for b in val_loader:
                imgs, labels = b["image"].to(device), b["label"].to(device)
                with autocast(enabled=amp):
                    val_logit = sliding_window_inference(imgs, roi_size, sw_bs, net)
                    vloss = loss_fun(val_logit, labels)
                val_loss_sum += vloss.item()
                #for metric: softmax -> argmax -> argmax label
                val_prob = torch.softmax(val_logit, dim=1) # [B, 2, D, H, W]
                val_cls = torch.argmax(val_prob, dim=1) # [B, D, H, W]
                
                # one-hot both pred and gt
                pred_oh = torch.nn.functional.one_hot(val_cls, num_classes=2) # [B, D, H, W, 2]
                label_oh = torch.nn.functional.one_hot(labels.squeeze(1).long(), num_classes=2) # [B, D, H, W, 2]
                
                pred_oh = torch.permute(pred_oh, (0, 4, 1, 2, 3)).float() # [B, 2, D, H, W]
                label_oh = torch.permute(label_oh, (0, 4, 1, 2, 3)).float() # [B, 2, D, H, W]
                
                y_pred_list = list(torch.unbind(pred_oh, dim=0)) #each item [2, D, H, W]
                y_true_list = list(torch.unbind(label_oh, dim=0)) #each item [2, D, H, W]
            
                dice_metric(y_pred=y_pred_list, y=y_true_list)
                
                
        mean_dice = dice_metric.aggregate().item()
        val_loss = val_loss_sum / max(1, len(val_loader))
        
        save_ckpt(run_dir / "last.pt")
        #keep a clean weights-only file for export/use at inference time
        torch.save(net.state_dict(), run_dir / "last_weights.pt")
        if mean_dice > best_dice + 1e-6:
            best_dice = mean_dice
            save_ckpt(run_dir / "best.pt")
            torch.save(net.state_dict(), run_dir / "best_weights.pt")
        
        #step the scheduler on metric mean_dice
        sceduler.step(mean_dice)
        
        lr = opt.param_groups[0]["lr"]
        print(f"epoch {epoch}/{epoches}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, mean_dice: {mean_dice:.4f}, lr: {lr:.6f}")
        
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/metric/mean_dice", mean_dice, epoch)
        writer.add_scalar("opt/lr", lr, epoch)
    writer.close()
        
        
if __name__ == "__main__":
    main()
    