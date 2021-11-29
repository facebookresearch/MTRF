# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

from sensor_msgs.msg import Image as Image_msg
import rospy
import numpy as np
import skimage
import matplotlib.pyplot as plt

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

class LogitechImageService(object):
    def __init__(self, image_shape=(32, 32, 3), topic="/front_1/image_raw"):
        self.image_shape = image_shape
        self.image = None
        # rospy.init_node('images_service', anonymous=True)
        print("subscribing to: ", topic)
        rospy.Subscriber(
            topic,
            Image_msg,
            self.store_image,
            queue_size=1,
            buff_size=2**24,
            tcp_nodelay=True)

        connection_attempts = 5
        for i in range(connection_attempts):
            if self.image is not None:
                break
            print("No image found yet.")
            rospy.sleep(1)

        if i == (connection_attempts - 1):
            raise ValueError

    def process_image(self, image):
        # image = np.flip(image.reshape((1080, 1920, 3)), axis=2)
        # image = image[50:950, 500:1400]

        image = image.reshape((480, 640, 3))
        # image = np.flip(image.reshape((480, 640, 3)), axis=2)
        # image = image[:, 80:-80, :]

        # image = np.flip(image.reshape((424, 512, 3)), axis=2)
        # image = image[:, 44:-44]

        # resize_to = next(
        #     2 ** i for i in reversed(range(10))
        #     if 2 ** i < image.shape[0])

        # image = skimage.transform.resize(
        #     image, (image.shape[0] // 2, image.shape[1] // 2), anti_aliasing=True, mode='constant')
        image = skimage.util.img_as_ubyte(image)

        # width = image.shape[0] // self.image_shape[0]
        # height = image.shape[1] // self.image_shape[1]
        # image = skimage.transform.downscale_local_mean(
        #     image, (width, height, 1))
        return image

    def store_image(self, data):
        # start = rospy.Time.now()

        image = np.frombuffer(data.data, dtype=np.uint8)
        image = self.process_image(image.copy())
        self.image = image

        # end = rospy.Time.now()

        # transport_delay = (start - data.header.stamp).to_sec()
        # process_delay = (end - start).to_sec()
        # total_delay = transport_delay + process_delay

        # print(f"Processing frame"
        #       f" | Transport delay:{transport_delay:6.3f}"
        #       f" | Process delay: {process_delay:6.3f}"
        #       f" | Total delay: {total_delay:6.3f}")

    def get_image(self, *args, width=32, height=32, **kwargs):
        # assert self.image.shape[:2] == (width, height)

        # old_image = self.image.copy()

        # same_frame = np.all(old_image == self._image)
        # if same_frame:
        #     self._same_frame_count += 1
        # else:
        #     self._same_frame_count = 0

        # if self._same_frame_count > 100:

        return self.image


def test_image_service():
    image_service = LogitechImageService()
    for i in range(10):
        image = image_service.get_image()
        if image is None:
            print("No pixels received yet")
        else:
            print(image.dtype, image.shape)

        show_images = False
        if show_images:
            plt.imshow(image.copy())
            plt.show()

        rospy.sleep(1)


if __name__ == '__main__':
    try:
        test_image_service()
    except rospy.ROSInterruptException:
        pass
