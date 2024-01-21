import torch
import torchvision.transforms.functional as tf


class ConditionalBrightness:
    def __init__(self, brightness_factor, threshold):
        """
        :param brightness_factor: Factor by which to increase the brightness.
        :param threshold: Brightness threshold below which the brightness adjustment is applied.
        """
        self.brightness_factor = brightness_factor
        self.threshold = threshold

    def __call__(self, img):
        """
        Apply conditional brightness adjustment while preserving completely black areas.
        :param img: Input image.
        :return: Brightened image.
        """
        # Convert image to PIL for processing
        pil_img = tf.to_pil_image(img)

        # Create a mask for non-black (non-zero) areas of the image
        non_black_mask = pil_img.convert("L").point(lambda x: 255 if x > 0 else 0)

        # Convert the non-black mask to a tensor and cast it to boolean
        non_black_mask_tensor = tf.to_tensor(non_black_mask).type(torch.bool)

        # Convert the image to a grayscale tensor for brightness calculation
        img_tensor = tf.to_tensor(pil_img.convert("L"))

        # Calculate the mean brightness of the non-black areas
        mean_brightness = img_tensor.masked_select(non_black_mask_tensor).float().mean().item()

        # Apply brightness adjustment if the mean brightness of non-black areas is below the threshold
        if mean_brightness < self.threshold:
            # Apply brightness adjustment
            brightened_img = tf.adjust_brightness(pil_img, self.brightness_factor)

            # Convert the brightened image to a tensor
            brightened_img_tensor = tf.to_tensor(brightened_img)

            # Combine the adjusted and original images using the mask
            final_img_tensor = torch.where(non_black_mask_tensor, brightened_img_tensor, img)
            final_img = tf.to_pil_image(final_img_tensor)
        else:
            final_img = pil_img

        return tf.to_tensor(final_img)
