from torchvision import io 


def read_image(path, image_read_mode = io.ImageReadMode.UNCHANGED):
    """
    path (str) – path of the JPEG or PNG image.

    image_read_mode (ImageReadMode) – the read mode used for optionally converting the image. 
                        Default: ImageReadMode.UNCHANGED. 

    Use ImageReadMode.UNCHANGED for loading the image as-is, 
    ImageReadMode.GRAY for converting to grayscale, 
    ImageReadMode.GRAY_ALPHA for grayscale with transparency, 
    ImageReadMode.RGB for RGB and 
    ImageReadMode.RGB_ALPHA for RGB with transparency.
    """
    return io.read_image(path, image_read_mode)



