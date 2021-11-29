# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import matplotlib.pyplot as plt
import numpy as np
from logitech_image_service import *

image_service = LogitechImageService(
    image_shape=(480, 640, 3),
    topic='/usb_cam/image_raw')

while input('q to exit: ') != 'q':
    image = image_service.get_image()
    print(type(image), image.shape)
    plt.imshow(image)
    import imageio
    imageio.imwrite('./dsafasdf.png', image)
    plt.show()

# sample_frequency = 1e-1
# num_samples = 100

# image_buffer = []
# import time
# import skvideo.io

# vid_num = 0
# while input('q to exit: ') != 'q':
#     for _ in range(num_samples):
#         image_buffer.append(image_service.get_image())
#         time.sleep(sample_frequency)
#     skvideo.io.vwrite(f'./test/video_{vid_num}.mp4', np.array(image_buffer))
#     vid_num += 1
