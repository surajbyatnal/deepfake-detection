import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, val_transform

def get_dataloaders(root_dir,
                    batch_size=8,
                    num_workers=0,
                    img_size=224,
                    pin_memory=True):
    """
    Expects directory structure:
    root_dir/
      train/
        fake/
        real/
      valid/ (or val/)
        fake/
        real/
      test/
        fake/
        real/
    Returns: train_loader, val_loader, test_loader, class_to_idx
    """
    train_dir = os.path.join(root_dir, "train")
    # Support both 'valid' and 'val' folder names
    val_dir = os.path.join(root_dir, "valid")
    if not os.path.exists(val_dir):
        val_dir = os.path.join(root_dir, "val")
    test_dir  = os.path.join(root_dir, "test")

    train_t, val_t = get_transforms(img_size=img_size)

    train_ds = ImageFolder(train_dir, transform=train_t)
    val_ds   = ImageFolder(val_dir, transform=val_t)
    test_ds  = ImageFolder(test_dir, transform=val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, train_ds.class_to_idx