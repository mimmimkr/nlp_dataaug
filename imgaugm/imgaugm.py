import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib

path = 'imgaugm\\pics\\'
folder = 'pics'

import time
start_time = time.time()


#IMPORT IMG FILE NAMES
f = []
for (dirpath, dirnames, filenames) in os.walk('imgaugm\\pics\\'):
    f.extend(filenames)
    break
print(f)

def imgaugp(type, filename):
    imgaugp.counter += 1
    augcounter = imgaugp.counter

    image = imageio.imread(path+filename)
    #ia.imshow(image)
    augname = type+'im'
    if(type == 'rotate'):
        rotate=iaa.Affine(rotate=(-50, 30))
        augimg=rotate.augment_image(image)
    if(type == 'scale'):
        print('scale')
        scale_im=iaa.Affine(scale={"x": (1.5, 1.0), "y": (1.5, 1.0)})
        augimg=scale_im.augment_image(image)
    if(type == 'noise'):
        gaussian_noise=iaa.AdditiveGaussianNoise(10,20)
        augimg=gaussian_noise.augment_image(image)
    if(type == 'crop'):
        crop = iaa.Crop(percent=(0, 0.3)) # crop image
        augimg=crop.augment_image(image)
    if(type == 'shear'):
        shear = iaa.Affine(shear=(0,40))
        augimg=shear.augment_image(image)
    if(type == 'fliplr'):
        #flipping image horizontally
        flip_hr=iaa.Fliplr(p=1.0)
        augimg= flip_hr.augment_image(image)
    if(type == 'flipud'):
        flip_vr=iaa.Flipud(p=1.0)
        augimg= flip_vr.augment_image(image)
    if(type == 'contrast'):
        contrast=iaa.GammaContrast(gamma=2.0)
        augimg=contrast.augment_image(image)
    if(type == 'artistic'):
        artistic= iaa.Cartoon(blur_ksize=3, segmentation_size=1.0,saturation=2.0, edge_prevalence=1.0)
        augimg=artistic.augment_image(image)
    if(type == 'weather'):
        snow= iaa.FastSnowyLandscape(lightness_threshold=[128, 200], lightness_multiplier=(1.5, 3.5))
        augimg=snow.augment_image(image)
    
    #save img to file w name
    imageio.imwrite(os.path.join(path, "aug_"+str(augcounter)+augname+".png" ), augimg)

imgaugp.counter = 0

for x in range(0,4):
    for x in range(len(f)):
        imgaugp('rotate', f[x])
        imgaugp('scale', f[x])
        imgaugp('noise', f[x])
        imgaugp('shear', f[x])
        imgaugp('crop', f[x])
        imgaugp('fliplr', f[x])
        imgaugp('flipud', f[x])
        imgaugp('contrast', f[x])
        imgaugp('artistic', f[x])
        imgaugp('weather', f[x])


print("--- %s seconds ---" % (time.time() - start_time))
