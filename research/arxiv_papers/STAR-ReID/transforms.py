from PIL import Image
import random
import numpy as np
from mindspore.dataset.vision import Inter
from mindspore.dataset.transforms import Compose


class Random2DTranslation:
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        height (int): target height.
        width (int): target width.
        p (float): probability of performing this transformation. Default: 0.5.
        interpolation (mindspore.dataset.vision.Inter): interpolation method. Default: Inter.BILINEAR.
    """

    def __init__(self, height, width, p=0.5, interpolation=Inter.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if random.random() < self.p:
            return img.resize((self.width, self.height), resample=self.interpolation)

        new_width, new_height = int(round(self.width * 1.125)), int(
            round(self.height * 1.125)
        )
        resized_img = img.resize((new_width, new_height), resample=self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        cropped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return cropped_img


def apply_random_2d_translation(img, height, width, p=0.5):
    transform = Random2DTranslation(height, width, p)
    return transform(img)


if __name__ == "__main__":
    img = Image.open("example.jpg").convert("RGB")
    transformer = Random2DTranslation(height=224, width=224, p=0.5)
    transformed_img = transformer(img)
    transformed_img.show()  # For visual inspection
