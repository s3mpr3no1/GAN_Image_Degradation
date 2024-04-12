from numpy import expand_dims, zeros, ones, cov, trace, iscomplexobj, asarray
from numpy.random import randn, randint, seed, shuffle
from keras.datasets.cifar10 import load_data
from ArtData import load_data_art
import os

from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
#from keras.optimizers import Adam

import matplotlib.pyplot as plt
#%matplotlib inline

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
#from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding, LayerNormalization
from keras.layers import LeakyReLU, Conv2DTranspose, Embedding, Concatenate


from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
# from Unpickle import load_data_new



from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
# from keras.optimizers import RMSprop



class cGAN():
  """
  The dataset argument to the train() method is of the form [TrainX, Trainy]
  where pixel values are [-1,1]
  """

  def __init__(self, epochs=151, batch_size=100, sample_interval=15, input_data=None, working_dir=r"C:\Users", run_number=1, run_header=""):
    self.img_rows = 32
    self.img_cols = 32
    self.channels = 3
    self.nclasses = 10
    self.run_number = run_number
    self.img_shape = (self.img_rows, self.img_cols, self.channels)
    self.latent_dim = 100
    self.losslog = []
    self.epochs = epochs
    self.batch_size = batch_size
    self.sample_interval = sample_interval
    self.working_dir = working_dir
    self.in_shape=(32,32,3)
    self.fid_sample_count = 10000
    self.run_header = run_header

    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    # self.opt = Adam(learning_rate=0.0002, beta_1=0.5)
    self.generator = self.build_generator()
    self.discriminator = self.build_discriminator()

    # make weights in the discriminator not trainable
    self.discriminator.trainable = False
    # get noise and label inputs from generator model
    self.gen_noise, self.gen_label = self.generator.input
    # get image output from the generator model
    self.gen_output = self.generator.output
    # connect image output and label input from generator as inputs to discriminator
    self.gan_output = self.discriminator([self.gen_output, self.gen_label])
    # define gan model as taking noise and label and outputting a classification
    self.model = Model([self.gen_noise, self.gen_label], self.gan_output)
    # compile model
    self.opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    # self.opt = Adam(learning_rate=0.0002, beta_1=0.5)
    self.model.compile(loss='binary_crossentropy', optimizer=opt)

    # Define the model for the FID score, we remove the top layer and re-add
    # the average pooling layer that was chopped off
    self.fid_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    # JAN
    (self.fid_data, _), (_, _) = load_data_art()
    shuffle(self.fid_data)
    self.fid_data = self.fid_data[:self.fid_sample_count]

    # Get the fid data ready for the calculate_fid function
    self.fid_data = self.fid_preprocess(self.fid_data)
    self.fid_log = []

    # Define the labels to be used for the FID calculation
    self.fid_labels = randint(0, self.nclasses, self.fid_sample_count).reshape(-1, 1)



  def build_generator(self):
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(self.nclasses, 50)(in_label)
    # linear multiplication
    n_nodes = 4 * 4
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((4, 4, 1))(li)
    # image generator input
    in_lat = Input(shape=(self.latent_dim,))
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((4, 4, 256))(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li])
    # upsample to 8x8
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
    gen = LeakyReLU(alpha=0.2)(gen)
    # upsample to 16x16
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # upsample to 32x32
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # output
    out_layer = Conv2D(3, (3,3), activation='tanh', padding='same')(gen)
    # define model
    model = Model([in_lat, in_label], out_layer)

    return model

  def build_discriminator(self):
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(self.nclasses, 50)(in_label)
    # scale up to image dimensions with linear activation
    n_nodes = self.in_shape[0] * self.in_shape[1]
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((self.in_shape[0], self.in_shape[1], 1))(li)
    # image input
    in_image = Input(shape=self.in_shape)
    # concat label as a channel
    merge = Concatenate()([in_image, li])


    # normal
    fe = Conv2D(64, (3,3), padding='same')(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample to 16x16
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample to 8x8
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample to 4x4
    fe = Conv2D(256, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # dropout
    fe = Dropout(0.4)(fe)
    # output
    out_layer = Dense(1, activation='sigmoid')(fe)


    # define model
    model = Model([in_image, in_label], out_layer)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model



  def load_real_samples(self):
    # load dataset
    (trainX, trainy), (_, _) = load_data_art()
    X = trainX
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return [X, trainy]



  def generate_real_samples(self, dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y

  # generate points in latent space as input for the generator
  def generate_latent_points(self, latent_dim, n_samples, n_classes=10):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


  # use the generator to generate n fake examples, with class labels
  def generate_fake_samples(self, generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = self.generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))
    return [images, labels_input], y


  def train(self, dataset):
    bat_per_epo = int(dataset[0].shape[0] / self.batch_size)
    half_batch = int(self.batch_size / 2)
    # manually enumerate epochs
    for i in range(self.epochs):
      # enumerate batches over the training set
      for j in range(bat_per_epo):
        # get randomly selected 'real' samples
        [X_real, labels_real], y_real = self.generate_real_samples(dataset, half_batch)
        # update discriminator model weights
        d_loss1, _ = self.discriminator.train_on_batch([X_real, labels_real], y_real)
        # generate 'fake' examples
        [X_fake, labels], y_fake = self.generate_fake_samples(self.generator, self.latent_dim, half_batch)
        # update discriminator model weights
        d_loss2, _ = self.discriminator.train_on_batch([X_fake, labels], y_fake)
        # prepare points in latent space as input for the generator
        [z_input, labels_input] = self.generate_latent_points(self.latent_dim, self.batch_size)
        # create inverted labels for the fake samples
        y_gan = ones((self.batch_size, 1))
        # update the generator via the discriminator's error
        g_loss = self.model.train_on_batch([z_input, labels_input], y_gan)
        # summarize loss on this batch
        print('Run %d >%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
        (self.run_number, i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
      if i % self.sample_interval == 0 and i >= 50:
        print("Generating performance summary")
        # Generate a plot
        # self.sample_images(i)

        # Calculate the FID score
        fid_images = self.sample_fid_images()
        FID = self.calculate_fid(self.fid_model, self.fid_data, fid_images)
        #self.fid_log.append((i, FID))
        # TODO: Write the FID log to a text file
        with open(os.path.join(self.working_dir, (self.run_header +"_FID_LOG.txt")), "a") as f:
            f.write(str(i)+":"+str(FID)+"\n")
        # self.generator.save(self.working_dir+ self.run_header + "_epoch_" + str(i)+ '.h5')

    # save the generator model
    #self.generator.save(self.working_dir+'cgan_generator.h5')




  def sample_fid_images(self):
    noise = np.random.normal(0, 1, (self.fid_sample_count, self.latent_dim))
    gen_imgs = self.generator.predict([noise, self.fid_labels])
    return self.fid_preprocess(gen_imgs, from_generator=True)


  def sample_images(self, epoch):
    
    noise = np.random.normal(0, 1, (10 * 10, self.latent_dim))
    # Makes labels 0-9 10 times
    sampled_labels = np.array(list(range(10))*10).reshape(-1, 1)


    # the shape is (100, 32, 32, 3)
    gen_imgs = self.generator.predict([noise, sampled_labels])


    gen_imgs = self.combine_images(gen_imgs)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * (gen_imgs + 1)

    plt.imshow(gen_imgs)
    plt.axis('off')
    plt.savefig(os.path.join(self.working_dir, (self.run_header + "_epoch_" + str(epoch)+ '.png')))
    plt.close()

  def combine_images(self, generated_images):
    # The shape of generated_images is 100, 32, 32, 3
    # The number of images
    num = generated_images.shape[0]

    # width = 10
    width = int(math.sqrt(num))

    # height = 10
    height = int(math.ceil(float(num)/width))

    
    shape = generated_images.shape[1:4]

    
    image = np.zeros((height*shape[0], width*shape[1], shape[2]),
                      dtype=generated_images.dtype)

    # enumerate returns the index and the object at each position
    for index, img in enumerate(generated_images):
        # int() does the floor operation
        # i below is the row number
        i = int(index/width)
        # j is the column number
        j = index % width
        # Added in the code to make it a large three-channel image
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1], 0:3] = \
            img[:, :, :]
    return image


  # Following methods are for the FID calculation
  # scale an array of images to a new size
  def scale_images(self, images, new_shape):
    images_list = list()
    for image in images:
      # resize with nearest neighbor interpolation
      new_image = resize(image, new_shape, 0)
      # store
      images_list.append(new_image)
    return asarray(images_list)


  # calculate frechet inception distance
  def calculate_fid(self, model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
      covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


  def fid_preprocess(self, images, from_generator=False):
    if from_generator:
        # Data coming from the generator is [-1,1]
        # But data from load_data is [0,255]
        images = (images * 127.5) + 127.5
        # This line puts images back to [0,255] like the rest

    # Convert to float
    images = images.astype('float32')
    # resize images
    images = self.scale_images(images, (299,299,3))
    # Images are [0,255] after scaling
    # pre-process images
    images = preprocess_input(images)
    # Images are [-1,1] after preprocessing
    return images


def main():
    # load dataset
    (trainX, trainy), (_, _) = load_data_art()
    X = trainX
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    regular_data = [X, trainy]

    test = cGAN()
    test.train(regular_data)

if __name__ == "__main__":
    main()





