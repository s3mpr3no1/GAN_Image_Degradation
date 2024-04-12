from cGAN_BigBatch import cGAN
from ImageDegradations import ImageDegrader
import pandas as pd
import numpy as np
import random as python_random
import tensorflow as tf
from keras.datasets.cifar10 import load_data
from Unpickle import load_data_new
from ArtData import load_data_art

np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(123)

"""
The goal of this code is to run 5 replicates at each level from 0% to 100% of the dataset featuring a given single corruption. The same partitions will be used for all iterations to standardize.

Changed the ImageDegrader class to initialize once and then take the dataset as an argument to load_and_degrade_set(). The data should be an iterable: [X, y]
"""

# ENTER WORKING DIRECTORY
working_dir = r"C:\Users"

(X, y), (_, _) = load_data_art()
dataset = [X, y]



degradations = ["rectangles", "brighten", "darken", "decrease_contrast", "increase_contrast", "gaussian_blur", "gaussian_noise", "jpeg_compression", "decolorize_grey", "removal"]



seeds = [314, 42, 21, 666, 123]

# All indices for the Cifar 10 dataset
indices = list(range(X.shape[0]))

# total_runs = len(degradations) * len(seeds) * 10
current_run = 0

Degrader = ImageDegrader()

# This is the outermost loop - each deg is one line on the graph
for degradation in degradations:
    
    for level in range(90, 91, 15):
        clean_indices = list(range(X.shape[0]))
        degraded_indices = []

        # Create the dataset
        workingX = np.copy(X)
        workingy = np.copy(y)

        degraded_indices = python_random.sample(clean_indices, int(5000 * (level / 10)))
        clean_indices = [x for x in clean_indices if x not in degraded_indices]

        if degradation != "removal":
            degraded_set = Degrader.load_and_degrade_set(input_data=[workingX, workingy], input_indices=degraded_indices, degradations=[degradation])
        else:
            shrunken_X = np.asarray([workingX[i] for i in range(len(workingX)) if i not in degraded_indices])
            shrunken_y = np.asarray([workingy[i] for i in range(len(workingy)) if i not in degraded_indices])
            degraded_set = Degrader.load_and_degrade_set(input_data=[shrunken_X, shrunken_y], input_indices=degraded_indices, degradations=[])

        # Each level has 5 replicates
        for replicate_seed in seeds:
            np.random.seed(replicate_seed)
            python_random.seed(replicate_seed)
            tf.random.set_seed(replicate_seed)
            run_header = degradation + "_" + str(level) + "_" + str(replicate_seed)
            # train GAN
            GAN = cGAN(working_dir=working_dir, run_number=current_run, run_header=run_header)
            GAN.train(degraded_set)
            current_run += 1

        
