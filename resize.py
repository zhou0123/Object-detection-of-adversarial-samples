import sys
import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil

from pip import main
def _Resize(newsize, annodir=None):
    annolist = os.listdir(annodir)
    total = len(annolist)
    #对全部标注文件名遍历
    for num, f in enumerate(annolist):
        anno_path = os.path.join(annodir, f)
        #对图像标注信息进行修改，之后会说明vol.changeone()函数
        _changeone(anno_path, None, None, newsize)
        #显示进度条
        process = int(num*100 / total)
        s1 = "\r%d%%[%s%s]"%(process,"*"*process," "*(100-process))
        s2 = "\r%d%%[%s]"%(100,"*"*100)
        sys.stdout.write(s1)
        sys.stdout.flush()
    sys.stdout.write(s2)
    sys.stdout.flush()
    print('')
    print('Resize is complete!')
def _changeone(annofile, oldcls, newcls, newsize=None):
        if os.path.exists(annofile) == False:
            raise FileNotFoundError
        tree = ET.parse(annofile)
        root = tree.getroot()
        annos = [anno for anno in root.iter()]
        for i, anno in enumerate(annos):
            if newsize != None:
                if 'width' in anno.tag:
                    oldwidth = float(anno.text)
                    anno.text = str(newsize[0])
                    sizechangerate_x = newsize[0] / oldwidth
                if 'height' in anno.tag:
                    oldheight = float(anno.text)
                    anno.text = str(newsize[1])
                    sizechangerate_y = newsize[1] / oldheight

            if 'object' in anno.tag:
                for element in list(anno):
                    if oldcls != newcls:
                        if 'name' in element.tag:
                            if element.text == oldcls:
                                element.text = newcls
                                print(os.path.basename(annofile)+' change the class name')
                        break
                    if newsize != None:
                        if 'bndbox' in element.tag:
                            for coordinate in list(element):
                                if 'xmin' in coordinate.tag:
                                    coordinate.text = str(int(int(coordinate.text) * sizechangerate_x))
                                if 'xmax' in coordinate.tag:
                                    coordinate.text = str(int(int(coordinate.text) * sizechangerate_x))
                                if 'ymin' in coordinate.tag:
                                    coordinate.text = str(int(int(coordinate.text) * sizechangerate_y))
                                if 'ymax' in coordinate.tag:
                                    coordinate.text = str(int(int(coordinate.text) * sizechangerate_y))
                        
        tree = ET.ElementTree(root)
        tree.write(annofile, encoding="utf-8", xml_declaration=True)
if __name__ == '__main__':
    _Resize([300,300],"/home/zcy/python/VOCdevkit/VOC2007/Annotations")
