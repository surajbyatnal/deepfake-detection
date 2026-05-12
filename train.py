import os
import sys

# Ensure project root is on sys.path so imports like `data` and `model` work
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import argparse
import time
import torch
import torch.nn as nn
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None
from torch.cuda.amp import GradScaler, autocast
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from data.data import get_dataloaders
from model.vit_cnn_model import HybridModel
from Evaluvate.Evaluvate_module import evaluate_logits


def save_checkpoint(state, path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    torch.save(state, path)

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, epoch=0):
    model.train()
    running_loss = 0.0
    total = 0
    
    if tqdm is not None:
        loader = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1).float()  # BCEWithLogits expects float shape (B,1)

        optimizer.zero_grad()
        if scaler is not None:
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
        
        # Print progress every 10 batches if no tqdm
        if tqdm is None and (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)}, Loss: {running_loss/total:.4f}")

    return running_loss / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())
    logits_all = torch.cat(logits_list, dim=0).squeeze(1)
    labels_all = torch.cat(labels_list, dim=0).squeeze(1)
    metrics = evaluate_logits(logits_all.numpy(), labels_all.numpy())
    metrics['val_loss'] = total_loss / (len(loader.dataset))
    return metrics

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        args.data_root, batch_size=args.batch_size, num_workers=args.num_workers, img_size=args.img_size, pin_memory=(device.type == 'cuda'))

    # Basic sanity checks
    print(f"Using device: {device}")
    print("Classes:", class_to_idx)
    if len(train_loader.dataset) == 0:
        raise RuntimeError(f"Train dataset appears empty: {os.path.join(args.data_root, 'train')}")
    if len(val_loader.dataset) == 0:
        raise RuntimeError(f"Validation dataset appears empty: {os.path.join(args.data_root, 'valid')}")

    # Create model with CLI options (both CNN and ViT pretrained by default)
    cnn_backbone = getattr(args, 'cnn_backbone', 'resnet50')
    cnn_pretrained_flag = not getattr(args, 'no_cnn_pretrained', False)
    vit_pretrained_flag = not getattr(args, 'no_vit_pretrained', False)
    freeze_backbones_flag = getattr(args, 'freeze_backbones', False)
    
    model = HybridModel(
        cnn_backbone=cnn_backbone,
        cnn_pretrained=cnn_pretrained_flag,
        vit_pretrained=vit_pretrained_flag,
        freeze_backbones=freeze_backbones_flag
    ).to(device)
    
    # Log model configuration
    total_params, trainable_params = model.count_parameters()
    print(f"Model: CNN={cnn_backbone}, CNN_pretrained={cnn_pretrained_flag}, ViT_pretrained={vit_pretrained_flag}")
    print(f"Freeze_backbones={freeze_backbones_flag}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    scaler = GradScaler() if (device.type == 'cuda' and args.use_amp) else None

    writer = None
    if SummaryWriter and args.log_dir:
        writer = SummaryWriter(log_dir=args.log_dir)

    best_auc = -1.0
    start_epoch = 0

    if args.resume:
        if not os.path.exists(args.resume):
            print(f"Warning: resume checkpoint {args.resume} not found, starting fresh")
        else:
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            opt_state = ckpt.get('opt_state', None)
            if opt_state:
                try:
                    optimizer.load_state_dict(opt_state)
                except Exception:
                    print("Warning: failed to load optimizer state from checkpoint; continuing without it")
            start_epoch = ckpt.get('epoch', 0) + 1
            best_auc = ckpt.get('best_auc', best_auc)
            print(f"Resumed from {args.resume}, starting epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        print(f"\nStarting Epoch {epoch}/{args.epochs-1}...")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler, epoch=epoch)
        print(f"Validating...")
        metrics = validate(model, val_loader, criterion, device)
        val_auc = metrics.get('auc', 0.0)

        scheduler.step()

        if writer:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('val/loss', metrics['val_loss'], epoch)
            writer.add_scalar('val/auc', val_auc, epoch)
            writer.flush()

        print(f"Epoch {epoch}/{args.epochs-1}  train_loss={train_loss:.4f}  val_loss={metrics['val_loss']:.4f}  val_auc={val_auc:.4f}  time={(time.time()-t0):.1f}s")

        # checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch:03d}_auc_{val_auc:.4f}.pth')
        save_checkpoint({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': optimizer.state_dict(),
            'best_auc': best_auc,
        }, ckpt_path)

        # save best
        if val_auc > best_auc:
            best_auc = val_auc
            best_path = os.path.join(args.checkpoint_dir, 'best.pth')
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'opt_state': optimizer.state_dict(),
                'best_auc': best_auc,
            }, best_path)
            print(f"Saved new best model to {best_path} (auc={best_auc:.4f})")

    if writer:
        writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='dataset', help='root dataset dir')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)   # on Windows, 0 or 1 recommended
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint to resume')
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--use_amp', action='store_true', help='use mixed precision if cuda')
    # model options for easier CPU-friendly runs
    parser.add_argument('--cnn_backbone', type=str, default='resnet50', choices=['resnet50','resnet18'],
                        help='CNN backbone to use (resnet18 is lighter)')
    parser.add_argument('--no_cnn_pretrained', action='store_true', help='do not use pretrained weights for CNN')
    parser.add_argument('--no_vit_pretrained', action='store_true', help='do not use pretrained weights for ViT')
    parser.add_argument('--freeze_backbones', action='store_true', help='freeze CNN and ViT backbones')
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    main(args)