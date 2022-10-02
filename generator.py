import os
import numpy as np
import torch
from torchvision.utils import save_image
import pandas as pd
# import imageio.v2 as imageio
import cv2
# from PIL import Image
from net import *


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])



z_size = 512
img_width = 128
img_height = 128

model_path = 'VAEmodel_6layers.pkl'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_path = "../dataset/tsushima_yoshiko/"
# img_list = pd.read_csv(img_path+'annotations.csv')
# img_num = img_list['image'].size

img_list = [x for x in os.listdir(img_path) if is_image_file(x)]

vae = VAE(zsize=z_size, layer_count=6)
vae.load_state_dict(torch.load(model_path, map_location=DEVICE))

# for i in range(500, 520):
for i in img_list:
    # img = cv2.imread(img_path+img_list['image'][i])
    img = cv2.imread(img_path+i)
    if img is None:
        continue
    img = cv2.resize(img, (img_width, img_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Image.fromarray(img).save('test.jpg')
    x = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose((2, 0, 1))) / 127.5 - 1.
    x = torch.unsqueeze(x, 0)
    # print(x.shape)

    x_rec, _, _ = vae(x)
    resultsample = torch.cat([x, x_rec]) * 0.5 + 0.5
    resultsample = resultsample.to(DEVICE)
    # save_image(resultsample.view(-1, 3, img_height, img_width),
    #             'results_rec/sample_' + "_" + str(i) + '.png')
    save_image(resultsample.view(-1, 3, img_height, img_width),
                'results_rec/'+ i + '.png')




