# flickr30k_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import json
from collections import defaultdict
from torchvision import transforms

class Flickr30kDataset(Dataset):
    """
    PyTorch Dataset for Flickr30k
    Each item returns:
      - image: preprocessed tensor
      - captions: list of reference captions
    """
    def __init__(self, image_dir, jsonl_path, transform=None):
        """
        Args:
            image_dir (str or Path): 图片文件夹路径
            jsonl_path (str or Path): ln_flickr30k_val_captions_multi.jsonl 文件路径
            transform: torchvision transforms to apply on image
        """
        self.image_dir = Path(image_dir)
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        # 读取 jsonl 并聚合 captions
        self.refs = defaultdict(list)  # image_id -> list of captions
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                img_id = obj["image_id"]   # 例如 "4960617393"
                caption = obj["caption"].strip()
                if caption:
                    self.refs[img_id].append(caption)

        # 保存 image_id 列表
        self.image_ids = list(self.refs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        captions = self.refs[img_id]
        # 构造图片路径
        img_path = self.image_dir / f"{img_id}.jpg"
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "captions": captions,
            "image_id": img_id
        }

# ----------------------------
# 使用示例
# ----------------------------
if __name__ == "__main__":
    dataset = Flickr30kDataset(
        image_dir="/data2/xiaoxinyu/data/flickr30k_images/flickr30k_images/flickr30k_images",
        jsonl_path="/data2/xiaoxinyu/data/flickr30k_images/ln_flickr30k_val_captions_multi.jsonl"
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # 测试
    for batch in dataloader:
        images = batch["image"]       # (B, 3, 224, 224)
        captions = batch["captions"]  # list of list of str
        ids = batch["image_id"]
        print("batch image tensor shape:", images.shape)
        print("captions[0]:", captions[0])
        print("image_ids:", ids)
        break
