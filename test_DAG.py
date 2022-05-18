#-*-coding:utf-8-*-
from __future__ import division

import os
import xml.etree.ElementTree as ET

import cv2
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
# from wider import WIDER
from skimage import transform as sktsf
from torch.autograd import Variable
import time

import attacks
from data.dataset import inverse_normalize
from data.dataset import pytorch_normalze
from data.util import read_image
from model import FasterRCNNVGG16
from trainer import BRFasterRcnnTrainer
from utils import array_tool as at
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')

# VOC_BBOX_LABEL_NAMES = (
#     'n02691156',
#     'n02419796',
#     'n02131653',
#     'n02834778',
#     'n01503061',
#     'n02924116',
#     'n02958343',
#     'n02402425',
#     'n02084071',
#     'n02121808',
#     'n02503517',
#     'n02118333',
#     'n02510455',
#     'n02342885',
#     'n02374451',
#     'n02129165',
#     'n01674464',
#     'n02484322',
#     'n03790512',
#     'n02324045',
#     'n02509815',
#     'n02411705',
#     'n01726692',
#     'n02355227',
#     'n02129604',
#     'n04468005',
#     'n01662784',
#     'n04530566',
#     'n02062744',
#     'n02391049'
# )


data_dir = '/home/zcy/python/Data/VOCdevkit/VOC2007'
# data_dir = '/home/xingxing/liangsiyuan/data/video_dataset'
# attacker_path = '/home/xlsy/Documents/CVPR19/final results/weights/attack_12211147_2500.path'
attacker_path = 'checkpoints/10.path'
save_path_HOME = '/home/zcy/python/'
# save_path_HOME = '/home/xingxing/liangsiyuan/results/ssd_attack_video'



class VOCBboxDataset:
    """Bounding box dataset for PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`VOC_BBOX_LABEL_NAMES`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data.
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    """

    def __init__(self, data_dir, split='test',
                 use_difficult=False, return_difficult=False,
                 ):
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))
        # id_list_file = os.listdir(data_dir)
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES
        self.save_dir = save_path_HOME + '/frcnn/'
        self.save_dir_adv = save_path_HOME + '/JPEGImages/'
        self.save_dir_perturb = save_path_HOME + '/frcnn_perturb/'

    def __len__(self):
        return len(self.ids)

    def preprocess(self, img, min_size=300, max_size=448):
        """Preprocess an image for feature extraction.

        The length of the shorter edge is scaled to :obj:`self.min_size`.
        After the scaling, if the length of the longer edge is longer than
        :param min_size:
        :obj:`self.max_size`, the image is scaled to fit the longer edge
        to :obj:`self.max_size`.

        After resizing the image, the image is subtracted by a mean image value
        :obj:`self.mean`.

        Args:
            img (~numpy.ndarray): An image. This is in CHW and RGB format.
                The range of its value is :math:`[0, 255]`.
             (~numpy.ndarray): An image. This is in CHW and RGB format.
                The range of its value is :math:`[0, 255]`.

        Returns:
            ~numpy.ndarray:
            A preprocessed image.

        """
        C, H, W = img.shape
        max_size = 400
        scale1 = min_size / min(H, W)
        scale2 = max_size / max(H, W)
        scale = min(scale1, scale2)
        img = img / 255.
        try:
            # img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect')
            img = sktsf.resize(img, (C, 300, 300), mode='reflect')
        except:
            ipdb.set_trace()
        # both the longer and shorter should be less than
        # max_size and min_size
        normalize = pytorch_normalze
        return normalize(img)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        # print('id of img is:' + id_)
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            # if not self.use_difficult and int(obj.find('difficult').text) == 1:
            #     continue

            # difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(np.zeros(label.shape), dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)

        img = self.preprocess(img)
        img = torch.from_numpy(img)[None]
        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bbox, label, difficult, id_

    __getitem__ = get_example


def add_bbox(ax,bbox,label,score):
    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=2))

        caption = list()
        label_names = list(VOC_BBOX_LABEL_NAMES) + ['bg']
        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax


def img2jpg(img, img_suffix, quality):
    jpg_base = '/media/drive/ibug/300W_cropped/frcnn_adv_jpg/'
    img = img.transpose((1, 2, 0))
    img = Image.fromarray(img.astype('uint8'))
    if not os.path.exists(jpg_base):
        os.makedirs(jpg_base)
    jpg_path = jpg_base + img_suffix
    img.save(jpg_path, format='JPEG', subsampling=0, quality=quality)
    jpg_img = read_image(jpg_path)
    return jpg_img



if __name__ == '__main__':
    layer_idx = 20
    faster_rcnn = FasterRCNNVGG16()

    





