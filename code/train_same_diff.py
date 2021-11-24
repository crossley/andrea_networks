# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 12:27:50 2021

@author: mq20185996
"""

import os
from pathlib import Path
from PIL import Image

if __name__ == '__main__':
    
    data_dir = Path(r'D:\Andrea_NN\data\IMAGENET\Compressed')
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'validation')
    
    base = Image.new('RGB', (224, 224))
    base.show()
    
    im1 = Image.open(r'D:\Andrea_NN\data\IMAGENET\Compressed\train\train\n01440764\n01440764_18.JPEG')
    im1.show()
    
    im2 = Image.open(r'D:\Andrea_NN\data\IMAGENET\Compressed\train\train\n01440764\n01440764_36.JPEG')
    im2.show()
    
    im1.thumbnail((224 // 3, 224//3))
    base.paste(im1, (0,0))
    im2.thumbnail((224//3, 224//3))
    base.paste(im2, (224 - im2.size[0], 224 - im2.size[1]))
    base.show()