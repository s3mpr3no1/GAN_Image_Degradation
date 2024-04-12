from numpy.random import randn
from numpy.random import randint
from numpy.random import uniform
from numpy.random import seed
from keras.datasets.cifar10 import load_data
from matplotlib import pyplot
from PIL import Image as im
import numpy as np
from albumentations.augmentations.blur.transforms import GaussianBlur
from albumentations.augmentations.transforms import JpegCompression, GaussNoise
import math
from PIL import Image, ImageEnhance
# from Unpickle import load_data_new
from ArtData import load_data_art


class ImageDegrader():
  """
  This class loads CIFAR-10 upon initiation. You can then pass numerous degradation
  schemes of the form {"degradataion_name": <%of dataset>}
  Example:
  {"rectangles":50, "gaussian_blur":50}. The result of the load_and_degrade_set()
  method is a list of the form [X, y] which is the format accepted by the GAN
  """


  def __init__(self):
    # np.random.seed(seed)
    # load cifar10 dataset
    (self.X, self.y), (_, _) = load_data_art()
    self.dataset = []

    self.noise = GaussNoise(always_apply=True)
    self.blur = GaussianBlur(always_apply=True)
    self.jpeg = JpegCompression(always_apply=True)



  def blur_noise_jpeg(self, img, degradation="gaussian_blur"):
    """
    This function applies gaussian blur, gaussian noise, or jpeg compression
    to an image. Both input and output are uint8 [0,255]
    """
    image = np.copy(img)
    if degradation == "gaussian_blur":
        image = self.blur(image=image)["image"]
    elif degradation == "gaussian_noise":
        image = self.noise(image=image)["image"]
    elif degradation == "jpeg_compression":
        image = self.noise(image=image)["image"]
    return image



  def brightness_darkness(self, img, degradation = "darken"):
    """
    This function takes in an image and returns a copy either darkened or brightened
    Pixel values are uint8 from 0 - 255
    """

    image = np.copy(img)
    image = image.astype('float32')
    if degradation == "darken":
      image = np.clip(image-128, 0, 255)
    else:
      image = np.clip(image+128, 0, 255)


    return image


  def decolorization(self, img, color_channel = "greyscale"):
    """
    This function takes in an image and zeroes out one of its color channels.
    Acceptable arguments for color_channel are "red", "blue", "green". The default
    value of greyscale will turn the whole image into a greyscale image.

    Input and output are uint8 [0,255]
    """
    image = np.copy(img)

    if color_channel == "red":
      image[:,:,0] = 0
    elif color_channel == "green":
      image[:,:,1] = 0
    elif color_channel == "blue":
      image[:,:,2] == 0

    else:
       
        image = im.fromarray(image)

        # Convert to HSV model
        H, S, V = image.convert('HSV').split()

        # Halve the saturation
        S = S.point(lambda p: p//2)

        # Remerge the channels
        HSV = im.merge('HSV', (H,S,V))
        # Convert the image back to RGB
        image = HSV.convert('RGB')
        image = np.array(image)
    image = image.astype('uint8')


    return image


  def random_rectangles(self, img, sl = 0.02, sh = 0.4, r1 = 0.3):
    """
    Adds a random rectangle to the input image. First two dimensions of the image
    are the height and width
    sl and sh are the min and max areas respectively
    rl is the minimum aspect ratio allowed (rh is the reciprocal)
    https://github.com/zhunzhong07/Random-Erasing

    This method returns a copy of the image; it does not alter it directly
    In and Out are integer [0,255]
    """
    image = np.copy(img)
    
    for attempt in range(100):
      # Define the area of the whole image
      area = image.shape[0] * image.shape[1]
      # Define the randomly chosen target area (between sl and sh)
      target_area = uniform(sl, sh) * area
      # Define a random aspect ratio
      aspect_ratio = uniform(r1, 1.0/r1)

      # Calculate the height and width of the rectangle
      h = int(round(math.sqrt(target_area * aspect_ratio)))
      w = int(round(math.sqrt(target_area / aspect_ratio)))


      if w < image.shape[1] and h < image.shape[0]:
        x1 = randint(0, image.shape[0] - h)
        y1 = randint(0, image.shape[1] - w)
        if image.shape[2] == 3:
            image[x1:x1+h, y1:y1+w, 0] = 0
            image[x1:x1+h, y1:y1+w, 1] = 0
            image[x1:x1+h, y1:y1+w, 2] = 0
        else:
            image[x1:x1+h, y1:y1+w, 0] = 0
        return image
    return image


  def contrast(self, img, direction="increase"):
    """
    This function increases or decreases the contrast of an image
    """
    factor = 2
    image = np.copy(img)
    cont_obj = ImageEnhance.Contrast(Image.fromarray(image))
    if direction == "decrease":
      factor = 0.5

    degraded = np.array(cont_obj.enhance(factor))

    return degraded



  def load_and_degrade_set(self, degradations=[], seed=123, input_data=-1, input_indices=-1):
    """
    This function loads the CIFAR-10 dataset and returns a copy of it with
    the appropriate degradations.

    degradations = Dictionary {<degradation_name>: level, ...}
    example: {gaussian_blur: 25, decolorization: 10}

    KEYS MUST BE ONE OF THE FOLLOWING: "decolorize_grey", "decolorize_red",
    "decolorize_blue", "decolorize_green", "gaussian_blur", "gaussian_noise",
    "jpeg_compression", "contrast", "rectangles", "brighten", "darken"

    Input images should have integer pixel values [0,255]. All degradations
    also output this type. This function outputs images with pixels [-1,1]
    """
    

    if input_data == -1:
    # So we don't have to redownload for each run
        X = np.copy(self.X)
        y = np.copy(self.y)
    else:
        X = input_data[0]
        y = input_data[1]

    if input_indices == -1:
        indices = range(X.shape[0])
    else:
        indices = input_indices

    # Loop over each CIFAR-10 image
    for sample_index in indices:
      # Loop over each degradation
      for degradation in degradations:
        # If the degradation gets applied
        temp = randint(1,100)
        if True:
          # Call the corruptions package for certain degradations

          if degradation == "increase_contrast":
            X[sample_index] = self.contrast(X[sample_index], direction="increase")
          elif degradation == "decrease_contrast":
            X[sample_index] = self.contrast(X[sample_index], direction="decrease")
          elif degradation in ["gaussian_blur", "gaussian_noise", "jpeg_compression"]:
            X[sample_index] = self.blur_noise_jpeg(X[sample_index], degradation=degradation)
          elif degradation in ["decolorize_grey", "decolorize_red", "decolorize_green", "decolorize_blue"]:
            if degradation == "decolorize_grey":
              X[sample_index] = self.decolorization(X[sample_index], color_channel="greyscale")
            elif degradation == "decolorize_red":
              X[sample_index] = self.decolorization(X[sample_index], color_channel="red")
            elif degradation == "decolorize_green":
              X[sample_index] = self.decolorization(X[sample_index], color_channel="green")
            elif degradation == "decolorize_blue":
              X[sample_index] = self.decolorization(X[sample_index], color_channel="blue")
          elif degradation in ["brighten", "darken"]:
            X[sample_index] = self.brightness_darkness(X[sample_index], degradation=degradation)
          elif degradation == "rectangles":
            X[sample_index] = self.random_rectangles(X[sample_index])




    # convert from unsigned ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    self.dataset = np.copy(X)
    return [X, y]



  def display_data(self):
    """
    Displays the first 100 images from the dataset in a grid
    """
    if self.dataset == []:
      print("Run load_and_degrade_set() first!")
      return
    dataset = np.copy(self.dataset[:100])
    dataset = (dataset + 1) / 2.0
    # plot images
    n = 10
    for i in range(n * n):
      # define subplot
      pyplot.subplot(n, n, 1 + i)
      # turn off axis
      pyplot.axis('off')
      # plot raw pixel data
      pyplot.imshow(dataset[i])
