import os

from torch.utils.data import Dataset
from PIL import Image


def scale_bb(target, w_2, h_2):
    scale_x = (w_2 / target[0]['w'])
    scale_y = (h_2 / target[0]['h'])
    for t in target:
        t['bbox'][0] *= scale_x
        t['bbox'][1] *= scale_y
        t['bbox'][2] *= scale_x
        t['bbox'][3] *= scale_y
    return target


class PascalVOC(Dataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.classes = ["aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair",
                        "cow", "diningtable", "dog", "horse",
                        "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train", "tvmonitor"]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        w = img.size[0]
        h = img.size[1]
        target[0]['w'] = w
        target[0]['h'] = h
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.ids)
