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

class KinectImageService(object):
    def __init__(self, image_shape=(32, 32, 3), topic="/kinect2/qhd/image_color"):
        self.image_shape = image_shape
        self.image = None
        rospy.init_node('images_service', anonymous=True)
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
        image = np.flip(image.reshape((540, 960, 3)), axis=2)
        image = image[:, 225:225+540]  # Free screw
        # image = image[:, 110:110+540]  # Fixed screw
        # image = image[84:422, :422]

        resize_to = next(
            2 ** i for i in reversed(range(10))
            if 2 ** i < image.shape[0])
        image = skimage.transform.resize(
            image, (resize_to, resize_to), anti_aliasing=True, mode='constant')
        width = image.shape[0] // self.image_shape[0]
        height = image.shape[1] // self.image_shape[1]
        image = skimage.transform.downscale_local_mean(
            image, (width, height, 1))
        image = skimage.util.img_as_ubyte(image)

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
        if self.image.shape[:2] != (width, height):
            old_width, old_height = self.image.shape[:2]
            assert old_width >= width and old_height >= height, (
                f'{(old_width, old_height)} needs to be >= {(width, height)}')
            old_image = self.image.copy()
            # skimage requires the image be converted to float first
            float_img = skimage.util.img_as_float(old_image)
            assert old_width % width == 0 and old_height % height == 0
            width_factor = old_width // width
            height_factor = old_height // height
            downsampled = skimage.transform.downscale_local_mean(
                float_img, (width_factor, height_factor, 1))
            # Convert back to uint8
            downsampled = skimage.util.img_as_ubyte(downsampled)
            return downsampled

        # old_image = self.image.copy()

        # same_frame = np.all(old_image == self._image)
        # if same_frame:
        #     self._same_frame_count += 1
        # else:
        #     self._same_frame_count = 0

        # if self._same_frame_count > 100:
        assert self.image.shape[:2] == (width, height)
        return self.image


def test_image_service():
    image_service = KinectImageService()
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
