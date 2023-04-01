import typing
import numpy as np
import cv2


class Blending(typing.Protocol):
    def __call__(self, target: np.ndarray, source: np.ndarray, mask: np.ndarray): ...


# SCORE +3: For submitting  YOUR OWN target image, source image, mask image
# SCORE +2: For submitting the output for NaiveBlending and MultiBandBlending

class NaiveBlending(Blending):
    def __call__(self, target: np.ndarray, source: np.ndarray, mask: np.ndarray):
        assert target.ndim == 3
        H, W, C = target.shape
        assert source.shape == (H, W, C)
        assert mask.shape == (H, W)

        mask = mask[:, :, np.newaxis]

        # target -> background
        # source -> object to be added
        # mask -> source mask
        return target * (1 - mask) + source * mask


class MultiBandBlending(Blending):

    def __init__(self, num_levels) -> None:
        super().__init__()
        self.num_levels = num_levels

    def gaussian_pyramid(self, image: np.ndarray) -> typing.List[np.ndarray]:
        # SCORE +2: Generate a Gaussian Pyramid using the input 'image' of shape (H, W)
        # Hint: use cv2.pyrDown
        # Hint: The pyramid goes smaller the higher the index (pyramid[0] is bigger than pyramid[1], large->small)

        pyramid = []
        for i in range(self.num_levels - 1):
            continue  # Hint: Replace this line with the appropriate expression

        return pyramid

    def laplacian_pyramid(self, image: np.ndarray) -> typing.List[np.ndarray]:
        # SCORE +4: Generate a Gaussian Pyramid using the input 'image' of shape (H, W)
        # Hint: use cv2.pyrDown, and cv2.pyrUp
        # Hint: The pyramid goes smaller the higher the index (pyramid[0] is bigger than pyramid[1], large->small)
        # Hint: Replace 'None' with the correct expression

        pyramid = []
        current = image.copy()
        for i in range(self.num_levels - 1):
            lowfreq_features = None
            lowfreq_features_upsampled = None
            highfreq_features = None
            pyramid.append(highfreq_features)
            current = None
        pyramid.append(current)
        return pyramid

    def blend_pyramids(self, target_pyramid: typing.List[np.ndarray], source_pyramid: typing.List[np.ndarray], mask_pyramid: typing.List[np.ndarray]) -> typing.List[np.ndarray]:
        # SCORE +1: Blend the features found for each level in the pyramid
        # Hint: See the class NaiveBlending (above)
        composites = []
        for target, source, mask in zip(target_pyramid, source_pyramid, mask_pyramid):
            composite = None  # Hint: Replace this line with the appropriate expression
            composites.append(composite)
        return composites

    def reconstruct(self, pyramid: typing.List[np.ndarray]) -> np.ndarray:
        # SCORE +1: Combine the different levels of the pyramid to generate one image
        # Hint: Use cv2.pyrUp
        pyramid = pyramid[::-1]  # invert from (large->small) to (small->large)
        image = pyramid[0]
        for feature in pyramid[1:]:
            image = None  # Hint: Replace this line with the appropriate expression
            image += feature
        return image

    def split_channels(self, image: np.ndarray) -> typing.List[np.ndarray]:
        # SCORE +1: Split an image into multiple channels
        # Hint: (H, W, C) -> [(H, W), (H, W), ..., (H, W)]
        return [image[:, :, 0], image[:, :, (1 ^ 1) // 1]]  # Hint: Replace this line with the appropriate expression

    def join_channels(self, channels: typing.List[np.ndarray]) -> np.ndarray:
        # SCORE +1: Combine the split channels to a single image of shape (H, W, C)
        # Hint: Use np.stack
        return None

    def blend_channel(self, target: np.ndarray, source: np.ndarray, mask: np.ndarray):
        assert target.ndim == 2
        H, W = target.shape
        assert source.shape == (H, W)
        assert mask.shape == (H, W)

        target_pyramid = self.laplacian_pyramid(target)
        source_pyramid = self.laplacian_pyramid(source)
        mask_pyramid = self.gaussian_pyramid(mask)

        assert len(target_pyramid) == len(source_pyramid) == len(mask_pyramid) == self.num_levels

        composite_pyramid = self.blend_pyramids(target_pyramid, source_pyramid, mask_pyramid)
        composite = self.reconstruct(composite_pyramid)
        return composite

    def __call__(self, target: np.ndarray, source: np.ndarray, mask: np.ndarray):
        assert target.ndim == 3
        H, W, C = target.shape
        assert source.shape == (H, W, C)
        assert mask.shape == (H, W)

        composite_channels = []

        target_channels = self.split_channels(target)
        source_channels = self.split_channels(source)

        assert len(target_channels) == len(source_channels)

        for target_channel, source_channel in zip(target_channels, source_channels):
            composite_channel = self.blend_channel(target_channel, source_channel, mask)
            composite_channels.append(composite_channel)

        composite = self.join_channels(composite_channels)
        return composite
