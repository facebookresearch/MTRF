try:
    from r3l.utils.camera import logitech_image_service
except ImportError as e:
    print("Cannot import logitech_image_service")
try:
    from r3l.utils.camera import kinect_image_service
except ImportError as e:
    print("Cannot import kinect_image_service")

IMAGE_SERVICE = None

AVAILABLE_IMAGE_SERVICES = {
    "kinect": kinect_image_service.KinectImageService,
    "logitech": logitech_image_service.LogitechImageService,
}

def get_image_service(*args, device: str = "logitech", topic: str = "", **kwargs):
    global IMAGE_SERVICE
    if IMAGE_SERVICE is None:
        assert device in AVAILABLE_IMAGE_SERVICES, (
            "Device must be one of the following options:\n"
            f"{list(AVAILABLE_IMAGE_SERVICES.keys())}"
        )
        print(f"Creating new image service of type={device} with topic=\"{topic}\"")
        IMAGE_SERVICE = AVAILABLE_IMAGE_SERVICES[device](*args, topic=topic, **kwargs)
    else:
        print("Already exists an image service")
    return IMAGE_SERVICE

