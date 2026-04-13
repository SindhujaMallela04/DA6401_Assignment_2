"""Dataset skeleton for Oxford-IIIT Pet.
"""

#loading data
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms.functional as F
import torch 
import numpy as np
import os
import xml.etree.ElementTree as ET

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    
    def __init__(self, root, split='trainval', transform=None):
        self.dataset = OxfordIIITPet(root = root, split=split, target_types = ["category", "segmentation"], transform=transform, download=True)
        self.annotations_dir = os.path.join(root, "oxford-iiit-pet", "annotations", "xmls")

    def __len__(self):
        return len(self.dataset)
    
    def get_bbox_from_xml(self, idx):
        img_filename = self.dataset._images[idx].stem
        xml_path = os.path.join(self.annotations_dir, img_filename + ".xml")

        if not os.path.exists(xml_path):
            return None
            
        tree = ET.parse(xml_path)
        root = tree.getroot()        
        obj = root.find("object")
        bndbox = obj.find("bndbox")

        orig_w = float(root.find("size/width").text)
        orig_h = float(root.find("size/height").text)

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        scale_x = 224.0 / orig_w
        scale_y = 224.0 / orig_h

        xmin *= scale_x
        xmax *= scale_x
        ymin *= scale_y
        ymax *= scale_y

        x_centre = (xmin + xmax) / 2
        y_centre = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin        

        return torch.tensor([x_centre, y_centre, width, height], dtype = torch.float32)
        

    
    def get_bbox(self, mask):
        pos = torch.where(mask == 1)
        if len(pos[0])  == 0:
            return torch.tensor([0, 0, 0, 0], dtype = torch.float32)
        
        y_min = torch.min(pos[0])
        y_max = torch.max(pos[0])
        x_min = torch.min(pos[1])
        x_max = torch.max(pos[1])

        # H, W = mask.shape

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = (x_max - x_min)
        height = (y_max - y_min)

        return torch.tensor([x_center, y_center, width, height], dtype = torch.float32)

    def __getitem__(self, idx):
        image, (label, mask) = self.dataset[idx]

        #Resizing both image and mask to 224x224
        image = F.resize(image, (224, 224))
        mask = F.resize(mask, (224, 224), interpolation=F.InterpolationMode.NEAREST)

        #Converting image
        image = F.to_tensor(image)
        image = F.normalize (image, mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

        #Converting mask
        mask = torch.from_numpy(np.array(mask)).long()
        mask = mask - 1
        
        bbox = self.get_bbox_from_xml(idx)
        if bbox is None :        
            bbox = self.get_bbox(mask)
        #later normalize bbox = bbox / image_size
        return image, int(label), bbox, mask
    
    def get_num_classes(self):
        """Returns the number of classes in the dataset."""
        return len(self.dataset.classes)

#Checking
# dataset = OxfordIIITPetDataset(root="./data")

# img, label, bbox, mask = dataset[0]

# print(type(mask))
# print(mask.shape)
# print(mask.unique())
# print(bbox) 