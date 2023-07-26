import os
from pdf2image import convert_from_path
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Conv2D, Conv2DTranspose, Flatten, LeakyReLU
from keras.optimizers import Adam
import tensorflow.python.platform
import random
import math
import seaborn as sns
import PIL
from keras import layers
from keras.layers import Dense, Reshape, UpSampling2D, Conv2D, BatchNormalization
from keras.layers import LeakyReLU, Dropout, ZeroPadding2D, Flatten, Activation
import tqdm
import warnings


#Folder path containing the PDFs
data_folder = 'C:/Users/karrencp/Downloads/OneDrive_2023-06-28'
desired_size = (8 * 150, 6 * 150)



# # Check if cached data exists
if os.path.exists('hundred.pkl'):
    # Load cached data
    image_df = pd.read_pickle('hundred.pkl')
else:
    # Create an empty DataFrame to store the image data and paths
    image_data = []
    image_paths = []
    image_titles = []

    # Traverse through the subdirectories and process PDFs
    for root, directories, files in os.walk(data_folder):
        for file in files:
            # Check if the file is a PDF
            if file.endswith('.pdf'):
                # Construct the full path to the PDF
                pdf_path = os.path.join(root, file)

                # Convert PDF to PIL images using pdf2image
                images = convert_from_path(pdf_path)
                file_name = os.path.basename(file)

                # Iterate over the images and store the data and path
                for i, image in enumerate(images):
                    # Convert images to RGB color mode
                    # Resize the image
                    resized_image = image.resize(desired_size, resample=Image.BILINEAR)

                    # Append the image data and path to the respective lists
                    image_data.append(np.array(resized_image))
                    image_paths.append(pdf_path)
                    image_titles.append(file_name)

    # Create a DataFrame using the image data and paths
    image_df = pd.DataFrame({'Path': image_paths, 'Data': image_data, 'Titles': image_titles})

    # Cache the data
    image_df.to_pickle('hundred.pkl')
# Load cached data
image_df = pd.read_pickle('hundred.pkl')

#---------------------------------------------------------------------------------------------------------------------------


image_df['Data'] = image_df['Data'].apply(lambda x: x / 255.0)
# box_images = image_df[image_df['Titles'].str.contains('Box')]
pin_images = image_df[image_df['Titles'].str.contains('Pin')]
train_df, test_df = train_test_split(pin_images, test_size=0.6, random_state=42)
latent_dim = 100
output_shape = (900, 1200, 3)

#Building a Generator
generator = Sequential()
generator.add(Dense(180*240*3,activation="relu",input_dim=latent_dim))
generator.add(Reshape((180,240,3)))
generator.add(UpSampling2D())
generator.add(Conv2D(256,kernel_size=3,padding="same"))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Activation("relu"))
generator.add(UpSampling2D())
generator.add(Conv2D(256,kernel_size=3,padding="same"))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Activation("relu"))
generator.add(UpSampling2D())
generator.add(Conv2D(256,kernel_size=3,padding="same"))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Activation("relu"))
generator.add(UpSampling2D())
generator.add(Conv2D(128,kernel_size=3,padding="same"))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Activation("relu"))
generator.add(Conv2D(3,kernel_size=3,padding="same"))
generator.add(Activation("tanh"))

# #Creating a random seed and output from generator
# seed = tf.random.normal([1, latent_dim])
# Generated_Portrait = generator(seed, training=False)
# #Plotting the image output of generator without training 
# plt.imshow(Generated_Portrait[0, :, :, 0])
# plt.axis("off")

#Building a Discriminator
discriminator = Sequential()
discriminator.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(2880, 3840,3), padding="same"))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.25))
discriminator.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))
discriminator.add(BatchNormalization(momentum=0.8))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.25))
discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
discriminator.add(BatchNormalization(momentum=0.8))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.25))
discriminator.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
discriminator.add(BatchNormalization(momentum=0.8))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.25))
discriminator.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
discriminator.add(BatchNormalization(momentum=0.8))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.25))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation="sigmoid"))

discriminator.compile(loss='binary_crossentropy', optimizer=Adam())


epochs = 1500
batch_size = 4
learning_rate = 0.02
# gan = build_gan(generator, discriminator)
# print("--------------------")
# print(gan)
# print("--------------------")
# gan.summary()

gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer=Adam())

def preprocess_image(image):
    # Convert the image data to uint8 and scale it to [0, 255]
    image = (image * 255).astype(np.uint8)
    # Resize the image to the desired size
    image = Image.fromarray(image).resize((180, 240))
    # Convert the PIL image back to a numpy array and normalize it to [0, 1]
    image = np.array(image) / 255.0
    return image

def train_gan(generator, discriminator, gan, train_images, epochs, batch_size, latent_dim):
    # Generate real labels for the discriminator
    real_labels = np.ones((batch_size, 1))

    for epoch in range(epochs):
        # Get a random batch of real images
        real_batch = train_images.sample(n=batch_size)['Data'].values
        real_images = np.array([preprocess_image(image) for image in real_batch])
        real_images = tf.convert_to_tensor(real_images, dtype=tf.float32)

        # Generate a batch of fake images from random noise
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)

        # Train the discriminator on real and fake images
        discriminator_loss_real = discriminator.train_on_batch(real_images, real_labels)
        discriminator_loss_fake = discriminator.train_on_batch(fake_images, 1 - real_labels)
        discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

        # Train the generator
        generator_loss = gan.train_on_batch(noise, real_labels)

        print(f"Epoch: {epoch+1}/{epochs}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}")




train_gan(generator, discriminator, gan, train_df, epochs, batch_size, latent_dim)

# Function to generate and visualize images
def generate_and_visualize_images(generator, latent_dim, num_images_to_generate=10):
    # Generate random noise as input to the generator
    noise = np.random.normal(0, 1, (num_images_to_generate, latent_dim))
    
    # Generate images using the generator
    generated_images = generator.predict(noise)

    # Plot the generated images
    plt.figure(figsize=(1.8, 2.4))
    for i in range(num_images_to_generate):
        generated_image = generated_images[i]
        generated_image = (generated_image * 255).astype(np.uint8)
        plt.subplot(1, num_images_to_generate, i+1)
        plt.imshow(generated_image)
        plt.axis('off')
        plt.show()

# Call the function to generate and visualize images
num_images_to_generate = 25
generate_and_visualize_images(generator, latent_dim, num_images_to_generate)

#___________________________________commented out code____________

# def build_generator(output_shape, learning_rate=0.2):
#     generator_input = Input(shape=(output_shape))
#     print('Generator input shape:', generator_input)
#     units = (900 // 2) * (1200 // 2) * 3  

#     x = Dense(units)(generator_input)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Reshape((900 // 2, 1200 // 2, 3))(x)  
#     x = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same')(x)  
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same')(x)  
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', activation='sigmoid')(x)  

#     generator = Model(generator_input, x)
#     generator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate))
#     print("Generator Output Shape:", x.shape)  
#     return generator

# generator = build_generator(latent_dim, output_shape)

# def build_discriminator(image_shape, learning_rate=0.2):
#     """
#     Build the discriminator model.

#     Parameters:
#         image_shape (tuple): The shape of the input images (height, width, channels).

#     Returns:
#         keras.models.Model: The compiled discriminator model.
#     """
#     input_image = Input(shape=image_shape)
#     print("THIS IS THE INPUT IMAGE:", input_image)
#     x = Conv2D(32, (5, 5), strides=(2, 2), padding='same')(input_image)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Flatten()(x)
#     output = Dense(1, activation='sigmoid')(x)
#     discriminator = Model(input_image, output)
#     discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
#     return discriminator





# image_shape = (900, 1200, 3)  
# discriminator = build_discriminator(image_shape)

# discriminator.summary()

# Define the GAN architecture
# def build_gan(generator, discriminator):
#     gan_input = Input(shape=(latent_dim,))
#     generated_image = generator(gan_input)
#     gan_output = discriminator(generated_image)
#     gan = Model(gan_input, gan_output)
#     gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate))
#     return gan
# ________________________________ Kaggle Model________________

# import random
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import PIL
# from PIL import Image
# import tensorflow  as tf
# from keras import layers
# from keras.models import Sequential
# from keras.layers import Dense, Reshape, UpSampling2D, Conv2D, BatchNormalization
# from keras.layers import LeakyReLU, Dropout, ZeroPadding2D, Flatten, Activation
# from keras.optimizers import Adam
# import tqdm
# import os
# import warnings
# warnings.filterwarnings("ignore")
# #Settings
# sns.set(rc={"axes.facecolor":"#EDE9DE","figure.facecolor":"#D8CA7E"})

# for dirname, _, filenames in os.walk("C:/Users/karrencp/Downloads/OneDrive_2023-06-28"):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# np.random.seed(42)

# #Importing data
# data_path = "C:/Users/karrencp/Downloads/OneDrive_2023-06-28"
# batch_s = 64
# #Import as tf.Dataset
# data = tf.keras.preprocessing.image_dataset_from_directory(data_path, label_mode = None, image_size = (64,64), batch_size = batch_s)

# latent_dim = 100
# g_resolution=2

# #Building a Generator
# generator = Sequential()
# generator.add(Dense(4*4*256,activation="relu",input_dim=latent_dim))
# generator.add(Reshape((4,4,256)))
# generator.add(UpSampling2D())
# generator.add(Conv2D(256,kernel_size=3,padding="same"))
# generator.add(BatchNormalization(momentum=0.8))
# generator.add(Activation("relu"))
# generator.add(UpSampling2D())
# generator.add(Conv2D(256,kernel_size=3,padding="same"))
# generator.add(BatchNormalization(momentum=0.8))
# generator.add(Activation("relu"))
# generator.add(UpSampling2D())
# generator.add(Conv2D(256,kernel_size=3,padding="same"))#
# generator.add(BatchNormalization(momentum=0.8))
# generator.add(Activation("relu"))
# generator.add(UpSampling2D())
# generator.add(Conv2D(128,kernel_size=3,padding="same"))
# generator.add(BatchNormalization(momentum=0.8))
# generator.add(Activation("relu"))
# generator.add(Conv2D(3,kernel_size=3,padding="same"))
# generator.add(Activation("tanh"))

# #Creating a random seed and output from generator
# seed = tf.random.normal([1, latent_dim])
# Generated_Portrait = generator(seed, training=False)
# #Plotting the image output of generator without training 
# plt.imshow(Generated_Portrait[0, :, :, 0])
# plt.axis("off")

# #Building a Discriminator
# discriminator = Sequential()
# discriminator.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(64,64,3), padding="same"))
# discriminator.add(LeakyReLU(alpha=0.2))
# discriminator.add(Dropout(0.25))
# discriminator.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
# discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))
# discriminator.add(BatchNormalization(momentum=0.8))
# discriminator.add(LeakyReLU(alpha=0.2))
# discriminator.add(Dropout(0.25))
# discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
# discriminator.add(BatchNormalization(momentum=0.8))
# discriminator.add(LeakyReLU(alpha=0.2))
# discriminator.add(Dropout(0.25))
# discriminator.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
# discriminator.add(BatchNormalization(momentum=0.8))
# discriminator.add(LeakyReLU(alpha=0.2))
# discriminator.add(Dropout(0.25))
# discriminator.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
# discriminator.add(BatchNormalization(momentum=0.8))
# discriminator.add(LeakyReLU(alpha=0.2))
# discriminator.add(Dropout(0.25))
# discriminator.add(Flatten())
# discriminator.add(Dense(1, activation="sigmoid"))

# class GAN(tf.keras.Model):
#     def __init__(self, discriminator, generator, latent_dim):
#         super(GAN, self).__init__()
#         self.discriminator = discriminator
#         self.generator = generator
#         self.latent_dim = latent_dim

#     def compile(self, d_optimizer, g_optimizer, loss_fn):
#         super(GAN, self).compile()
#         self.d_optimizer = d_optimizer
#         self.g_optimizer = g_optimizer
#         self.loss_fn = loss_fn
#         self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
#         self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

#     @property
#     def metrics(self):
#         return [self.d_loss_metric, self.g_loss_metric]

#     def train_step(self, real_images):
#         # Sample random points in the latent space
#         batch_size = tf.shape(real_images)[0]
#         seed = tf.random.normal(shape=(batch_size, self.latent_dim))
#         # Decode them to fake images
#         generated_images = self.generator(seed)
#         # Combine them with real images
#         combined_images = tf.concat([generated_images, real_images], axis=0)
#         # Assemble labels discriminating real from fake images
#         labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
#         # Add random noise to the labels - important trick!
#         labels += 0.05 * tf.random.uniform(tf.shape(labels))
#         # Train the discriminator
#         with tf.GradientTape() as tape:
#             predictions = self.discriminator(combined_images)
#             d_loss = self.loss_fn(labels, predictions)
#         grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
#         self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

#         # Sample random points in the latent space
#         seed = tf.random.normal(shape=(batch_size, self.latent_dim))

#         # Assemble labels that say "all real images"
#         misleading_labels = tf.zeros((batch_size, 1))
#         with tf.GradientTape() as tape:
#             predictions = self.discriminator(self.generator(seed))
#             g_loss = self.loss_fn(misleading_labels, predictions)
#         grads = tape.gradient(g_loss, self.generator.trainable_weights)
#         self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

#         # Update metrics
#         self.d_loss_metric.update_state(d_loss)
#         self.g_loss_metric.update_state(g_loss)
#         return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}

# #Defining the number of epochs
# epochs = 2000
# #The optimizers for Generator and Discriminator
# discriminator_opt = tf.keras.optimizers.Adamax(1.5e-4,0.5)
# generator_opt = tf.keras.optimizers.Adamax(1.5e-4,0.5)
# #To compute cross entropy loss
# loss_fn = tf.keras.losses.BinaryCrossentropy()

# #Defining GAN Model
# model = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)

# #Compiling GAN Model
# model.compile(d_optimizer=discriminator_opt, g_optimizer=generator_opt, loss_fn=loss_fn)

# #Fitting the GAN
# history = model.fit(data, epochs=epochs)

# num_img=18

# #A function to generate and save images
# def Potrait_Generator():
#     Generated_Paintings = []
#     seed = tf.random.normal([num_img, latent_dim])
#     generated_image = generator(seed)
#     generated_image *= 255 
#     generated_image = generated_image.numpy()
#     for i in range(num_img):
#             img = tf.keras.preprocessing.image.array_to_img(generated_image[i])
#             Generated_Paintings.append(img)
#             img.save("Eng{:02d}.png".format(i)) 
#     return 

# #Generating images
# Images = Potrait_Generator()

# def train_gan(generator, discriminator, gan, train_images, epochs, batch_size, latent_dim):
#     # Generate real labels for the discriminator
#     real_labels = np.ones((batch_size, 1))

#     # Generate fake labels for the discriminator
#     fake_labels = np.zeros((batch_size, 1))
#     # print("------------------")
#     # print(len(train_images))
#     # print("------------------")
#     count = 0
#     for epoch in range(epochs):
#         # print("------------------")
#         # print("1")
#         # print("------------------")
#         for _ in range(len(train_images) // batch_size):
#             # print("------------------")
#             # print("2")
#             # print("------------------")
#             # Train the discriminator
#             # Get a random batch of real images
#             real_images = train_images.sample(n=batch_size)['Data'].values
#             real_images = np.array([np.array(image) for image in real_images])
#             real_images = tf.convert_to_tensor(real_images, dtype=tf.float32)

#             # resized_images = []
#             # for image in real_images:
#             #     image = image.numpy()
#             #     image = (image * 255).astype(np.uint8)
#             #     pil_image = Image.fromarray(image)
#             #     resized_image = pil_image.resize((400, 300))
#             #     resized_images.append(np.array(resized_image))
#             # real_images = tf.convert_to_tensor(resized_images, dtype=tf.float32)
#             # print("------------------")
#             # print("3")
#             # print("------------------")
#             # Generate a batch of fake images from random noise
#             noise = np.random.normal(0, 1, (batch_size, latent_dim))
#             fake_images = generator.predict(noise)
#             # print("------------------")
#             # print("4")
#             # print("------------------")
#             # Train the discriminator on real and fake images
#             discriminator_loss_real = discriminator.train_on_batch(real_images, real_labels)
#             discriminator_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
#             discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
#             # print("------------------")
#             # print("5")
#             # print("------------------")
#             generator_loss = gan.train_on_batch(noise, real_labels)
#             # print("------------------")
#             # print("6")
#             # print("------------------")
#         print(f"Epoch: {epoch+1}/{epochs}, Discriminator Loss: {discriminator_loss[0]}, Generator Loss: {generator_loss}")


# learning_rate = 0.0002
# generator = build_generator(latent_dim, learning_rate)
# discriminator = build_discriminator(image_shape, learning_rate)


# train_gan(generator, discriminator, gan, train_df, epochs, batch_size, latent_dim)



# # Function to generate and visualize images
# def generate_and_visualize_images(generator, latent_dim, num_images_to_generate=10):
#     # Generate random noise as input to the generator
#     noise = np.random.normal(0, 1, (num_images_to_generate, latent_dim))
    
#     # Generate images using the generator
#     generated_images = generator.predict(noise)

#     # Plot the generated images
#     plt.figure(figsize=(10, 10))
#     for i in range(num_images_to_generate):
#         generated_image = generated_images[i]
#         generated_image = (generated_image * 255).astype(np.uint8)
#         plt.subplot(1, num_images_to_generate, i+1)
#         plt.imshow(generated_image)
#         plt.axis('off')
# #     plt.show()

# # Call the function to generate and visualize images
# num_images_to_generate = 10
# generate_and_visualize_images(generator, latent_dim, num_images_to_generate)

# # ---------------------------------------------------------------------------------------------------------------------------------------

# import os
# from pdf2image import convert_from_path
# import pandas as pd
# import numpy as np
# from sklearn.cluster import DBSCAN
# from PIL import Image
# import joblib
# import matplotlib.pyplot as plt
# import cv2
# from sklearn.decomposition import PCA
# from sklearn.cluster import SpectralClustering
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, Reshape, Conv2D, Conv2DTranspose, Flatten
# from keras.optimizers import Adam
# import tensorflow.python.platform


# # Folder path containing the PDFs
# data_folder = 'C:/Users/karrencp/Downloads/OneDrive_2023-06-28'
# desired_size = (8 * 150, 6 * 150)

# # # Check if cached data exists
# if os.path.exists('hundred.pkl'):
#     # Load cached data
#     image_df = pd.read_pickle('hundred.pkl')
# else:
#     # Create an empty DataFrame to store the image data and paths
#     image_data = []
#     image_paths = []
#     image_titles = []

#     # Traverse through the subdirectories and process PDFs
#     for root, directories, files in os.walk(data_folder):
#         for file in files:
#             # Check if the file is a PDF
#             if file.endswith('.pdf'):
#                 # Construct the full path to the PDF
#                 pdf_path = os.path.join(root, file)

#                 # Convert PDF to PIL images using pdf2image
#                 images = convert_from_path(pdf_path)
#                 file_name = os.path.basename(file)

#                 # Iterate over the images and store the data and path
#                 for i, image in enumerate(images):
#                     # Convert images to RGB color mode
#                     # Resize the image
#                     resized_image = image.resize(desired_size, resample=Image.BILINEAR)

#                     # Append the image data and path to the respective lists
#                     image_data.append(np.array(resized_image))
#                     image_paths.append(pdf_path)
#                     image_titles.append(file_name)

#     # Create a DataFrame using the image data and paths
#     image_df = pd.DataFrame({'Path': image_paths, 'Data': image_data, 'Titles': image_titles})

#     # Cache the data
#     image_df.to_pickle('hundred.pkl')
# # Load cached data
# image_df = pd.read_pickle('hundred.pkl')

# # Convert the images to RGB color mode
# image_df['Data'] = image_df['Data'].apply(lambda x: Image.fromarray(x).convert('RGB'))
# box_pin_images = image_df[image_df['Titles'].str.contains('Box|Pin')]
# train_df, test_df = train_test_split(box_pin_images, test_size=0.2, random_state=42)
# image_height, image_width = image_df['Data'].iloc[0].size
# image_channels = 3



# latent_dim = 100  # Example value, adjust according to your needs
# # Define the generator architecture
# generator = Sequential([
#     Dense(7 * 7 * 256, activation='relu', input_shape=(latent_dim,)),
#     Reshape((7, 7, 256)),
#     Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu'),
#     Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
#     Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid')
# ])
# discriminator = Sequential([
#     Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(image_height, image_width, image_channels)),
#     Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu'),
#     Flatten(),
#     Dense(1, activation='sigmoid')
# ])

# # Get the output shape of the generator
# generator_output_shape = generator.output_shape[1:]  # Shape without the batch dimension

# # Update the discriminator input shape
# discriminator_input_shape = generator_output_shape + (image_channels,)

# # Define the new discriminator architecture with updated input shape
# discriminator = Sequential([
#     Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=generator_output_shape),
#     Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu'),
#     Flatten(),
#     Dense(1, activation='sigmoid')
# ])
# # Compile the discriminator
# discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# # Compile the generator
# generator.compile(loss='binary_crossentropy', optimizer=Adam())

# # Define the GAN architecture
# gan = Sequential([generator, discriminator])

# # Compile the GAN
# gan.compile(loss='binary_crossentropy', optimizer=Adam())

# # Define the training loop
# def train_gan(train_images, epochs, batch_size, latent_dim):
#     # Generate real labels for the discriminator
#     real_labels = np.ones((batch_size, 1))

#     # Generate fake labels for the discriminator
#     fake_labels = np.zeros((batch_size, 1))

#     for epoch in range(epochs):
#         for _ in range(len(train_images) // batch_size):
#             # Train the discriminator
#             # Get a random batch of real images
#             real_images = train_images.sample(n=batch_size)['Data'].values

#             # Generate a batch of fake images from random noise
#             noise = np.random.normal(0, 1, (batch_size, latent_dim))
#             fake_images = generator.predict(noise)
#             # Train the discriminator on real and fake images
#             real_images = np.array([np.array(image) for image in real_images])
#             resized_images = []
#             for image in real_images:
#                 pil_image = Image.fromarray(image)
#                 pil_image = pil_image.convert('L')  # Convert to grayscale
#                 resized_image = pil_image.resize((28, 28))
#                 resized_images.append(np.array(resized_image))
#             real_images = tf.convert_to_tensor(resized_images, dtype=tf.float32)


#             discriminator_loss_real = discriminator.train_on_batch(real_images, real_labels)
#             discriminator_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
#             discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

#             # Train the generator
#             # Generate a batch of fake images from random noise
#             noise = np.random.normal(0, 1, (batch_size, latent_dim))
#             # Train the generator to fool the discriminator
#             generator_loss = gan.train_on_batch(noise, real_labels)

#             # Print the progress
#             print(f"Epoch: {epoch+1}/{epochs}, Discriminator Loss: {discriminator_loss[0]}, Generator Loss: {generator_loss}")

# # Train the GAN
# train_gan(box_pin_images, epochs=5000, batch_size=32, latent_dim=100)

# num_images_to_generate = 10  # Number of images to generate
# noise = np.random.normal(0, 1, (num_images_to_generate, latent_dim))
# generated_images = generator.predict(noise)

# for i in range(num_images_to_generate):
#     generated_image = generated_images[i]  # Get a generated image
#     # Perform any necessary post-processing or conversion (e.g., scaling, denormalization)
#     # Save the generated image
#     image_path = f"generated_image_{i}.png"
#     cv2.imwrite(image_path, generated_image)
#     generated_image = cv2.imread(image_path)
#     plt.imshow(generated_image)
#     plt.axis('off')
#     plt.show()
# ----------------------------------------------------------------------------------------------------------------------------------------


# print(image_df['Data'])
# for index, image_array in enumerate(image_df['Data']):
#     print(f"Image {index} shape: {image_array.shape}")
# Reshape image data to (n_samples, n_features)
# Reshape image data to (n_samples, n_features)
# # Reshape image data to (n_samples, n_features)
# n_samples = len(image_df)
# height, width, _ = image_df['Data'].iloc[0].shape
# n_features = height * width * 3
# X = np.stack(image_df['Data'].apply(lambda x: x.reshape(-1)).tolist())



# # Apply SpectralClustering
# spectral_clustering = SpectralClustering(n_clusters=2)
# cluster_labels = spectral_clustering.fit_predict(X)
# image_df['Cluster'] = cluster_labels
# purple_dots = image_df[image_df['Cluster'] == 0]
# yellow_dots = image_df[image_df['Cluster'] == 1]

# # Separate data into different groups
# # Separate data into different groups
# box_images = image_df[image_df['Titles'].str.contains('Box')]
# pin_images = image_df[image_df['Titles'].str.contains('Pin')]
# other_images = image_df[~image_df['Titles'].str.contains('Box|Pin')]

# # Apply SpectralClustering to box_images
# box_clustering = SpectralClustering(n_clusters=2)
# box_cluster_labels = box_clustering.fit_predict(np.stack(box_images['Data'].apply(lambda x: x.reshape(-1)).tolist()))

# # Apply SpectralClustering to pin_images
# pin_clustering = SpectralClustering(n_clusters=3)
# pin_cluster_labels = pin_clustering.fit_predict(np.stack(pin_images['Data'].apply(lambda x: x.reshape(-1)).tolist()))

# # Apply SpectralClustering to other_images
# other_clustering = SpectralClustering(n_clusters=4)
# other_cluster_labels = other_clustering.fit_predict(np.stack(other_images['Data'].apply(lambda x: x.reshape(-1)).tolist()))

# # Perform dimensionality reduction for visualization
# pca = PCA(n_components=2)
# box_data_pca = pca.fit_transform(np.stack(box_images['Data'].apply(lambda x: x.reshape(-1)).tolist()))
# pin_data_pca = pca.fit_transform(np.stack(pin_images['Data'].apply(lambda x: x.reshape(-1)).tolist()))
# other_data_pca = pca.fit_transform(np.stack(other_images['Data'].apply(lambda x: x.reshape(-1)).tolist()))

# # Plot scatter plots for all groups on one graph
# plt.scatter(box_data_pca[:, 0], box_data_pca[:, 1], c='purple', label='Box Images')
# plt.scatter(pin_data_pca[:, 0], pin_data_pca[:, 1], c='yellow', label='Pin Images')
# plt.scatter(other_data_pca[:, 0], other_data_pca[:, 1], c='blue', label='Other Images')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.title('Clustering Results')
# plt.legend()
# plt.show()



# Separate data into different groups
# box_images = image_df[image_df['Titles'].str.contains('Box')]
# pin_images = image_df[image_df['Titles'].str.contains('Pin')]
# other_images = image_df[~image_df['Titles'].str.contains('Box|Pin')]



# box_clustering = SpectralClustering(n_clusters=2)
# box_cluster_labels = box_clustering.fit_predict(np.stack(box_images['Data'].apply(lambda x: x.reshape(-1)).tolist()))


# box_pca = PCA(n_components=2)
# box_data_pca = box_pca.fit_transform(np.stack(box_images['Data'].apply(lambda x: x.reshape(-1)).tolist()))


# plt.scatter(box_data_pca[:, 0], box_data_pca[:, 1], c=box_cluster_labels)
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.title('Box Images Clustering')
# plt.show()


# pin_clustering = SpectralClustering(n_clusters=3)
# pin_cluster_labels = pin_clustering.fit_predict(np.stack(pin_images['Data'].apply(lambda x: x.reshape(-1)).tolist()))


# pin_pca = PCA(n_components=2)
# pin_data_pca = pin_pca.fit_transform(np.stack(pin_images['Data'].apply(lambda x: x.reshape(-1)).tolist()))


# plt.scatter(pin_data_pca[:, 0], pin_data_pca[:, 1], c=pin_cluster_labels)
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.title('Pin Images Clustering')
# plt.show()


# other_clustering = SpectralClustering(n_clusters=4)
# other_cluster_labels = other_clustering.fit_predict(np.stack(other_images['Data'].apply(lambda x: x.reshape(-1)).tolist()))


# other_pca = PCA(n_components=2)
# other_data_pca = other_pca.fit_transform(np.stack(other_images['Data'].apply(lambda x: x.reshape(-1)).tolist()))


# plt.scatter(other_data_pca[:, 0], other_data_pca[:, 1], c=other_cluster_labels)
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.title('Other Images Clustering')
# plt.show()

# import os
# from pdf2image import convert_from_path
# import pandas as pd
# import numpy as np
# from sklearn.cluster import DBSCAN
# from PIL import Image
# import joblib
# import matplotlib.pyplot as plt
# import cv2
# from sklearn.decomposition import PCA
# from sklearn.cluster import SpectralClustering


# # Folder path containing the PDFs
# data_folder = 'C:/Users/karrencp/Downloads/OneDrive_2023-06-28'
# desired_size = (8 * 150, 6 * 150)

# # # Check if cached data exists
# if os.path.exists('hundred.pkl'):
#     # Load cached data
#     image_df = pd.read_pickle('hundred.pkl')
# else:
#     #Create an empty DataFrame to store the image data and paths
#     image_data = []
#     image_paths = []
#     image_titles = []

#     # Traverse through the subdirectories and process PDFs
#     for root, directories, files in os.walk(data_folder):
#         for file in files:
#             # Check if the file is a PDF
#             if file.endswith('.pdf'):
#                 # Construct the full path to the PDF
#                 pdf_path = os.path.join(root, file)

#                 # Convert PDF to PIL images using pdf2image
#                 images = convert_from_path(pdf_path)
#                 file_name = os.path.basename(file)

#                 # Iterate over the images and store the data and path
#                 for i, image in enumerate(images):
#                     # Resize the image
#                     resized_image = image.resize(desired_size, resample=Image.BILINEAR)

#                     # Append the image data and path to the respective lists
#                     image_data.append(np.array(resized_image))
#                     image_paths.append(pdf_path)
#                     image_titles.append(file_name)

#     # Create a DataFrame using the image data and paths
#         image_df = pd.DataFrame({'Path': image_paths, 'Data': image_data, 'Titles': image_titles})

#     # Cache the data
#         image_df.to_pickle('hundred.pkl')
# # print(image_df['Data'])
# # for index, image_array in enumerate(image_df['Data']):
# #     print(f"Image {index} shape: {image_array.shape}")
# # Reshape image data to (n_samples, n_features)
# # Reshape image data to (n_samples, n_features)
# # Reshape image data to (n_samples, n_features)
# n_samples = len(image_df)
# height, width, _ = image_df['Data'].iloc[0].shape
# n_features = height * width * 3
# X = np.stack(image_df['Data'].apply(lambda x: x.reshape(-1)).tolist())



# # Add a new column called 'Title' and assign the titles list to it

# # Apply SpectralClustering
# spectral_clustering = SpectralClustering(n_clusters=2)
# cluster_labels = spectral_clustering.fit_predict(X)
# image_df['Cluster'] = cluster_labels
# purple_dots = image_df[image_df['Cluster'] == 0]
# yellow_dots = image_df[image_df['Cluster'] == 1]

# # Create separate dataframes for different groups
# box_images = image_df[image_df['Titles'].str.contains('Box')]
# pin_images = image_df[image_df['Titles'].str.contains('Pin')]
# other_images = image_df[~image_df['Titles'].str.contains('Box|Pin')]

# Print the titles of each group
# print("Box Images:")
# print(box_images['Titles'])
# print("\nPin Images:")
# print(pin_images['Titles'])
# print("\nOther Images:")
# print(other_images['Titles'])

# image_df['Title'] = titles
# print(image_df)
# print(yellow_dots)
# for index, row in yellow_dots.iterrows():
#     print(row['Path'])
# print(purple_dots)

# print(image_df.columns)
# print(image_df['Titles'])

# #Perform dimensionality reduction for visualization
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# # Plot the scatter plot
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels)
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.title('Spectral Clustering Results')
# plt.show()



# import os
# from pdf2image import convert_from_path
# import pandas as pd
# import numpy as np
# from sklearn.cluster import DBSCAN
# from PIL import Image
# import joblib
# import matplotlib.pyplot as plt
# import cv2
# from sklearn.decomposition import PCA
# from sklearn.cluster import SpectralClustering


# # Folder path containing the PDFs
# data_folder = 'C:/Users/karrencp/Downloads/OneDrive_2023-06-28'
# desired_size = (8 * 150, 6 * 150)

# # Check if cached data exists
# if os.path.exists('cached_data.pkl'):
#     # Load cached data
#     image_df = pd.read_pickle('cached_data.pkl')
# else:
#     # Create an empty DataFrame to store the image data and paths
#     image_data = []
#     image_paths = []

#     # Traverse through the subdirectories and process PDFs
#     for root, directories, files in os.walk(data_folder):
#         for file in files:
#             # Check if the file is a PDF
#             if file.endswith('.pdf'):
#                 # Construct the full path to the PDF
#                 pdf_path = os.path.join(root, file)

#                 # Convert PDF to PIL images using pdf2image
#                 images = convert_from_path(pdf_path)

#                 # Iterate over the images and store the data and path
#                 for i, image in enumerate(images):
#                     # Resize the image
#                     resized_image = image.resize(desired_size, resample=Image.BILINEAR)

#                     # Append the image data and path to the respective lists
#                     image_data.append(np.array(resized_image))
#                     image_paths.append(pdf_path)

#     # Create a DataFrame using the image data and paths
#     image_df = pd.DataFrame({'Path': image_paths, 'Data': image_data})

#     # Cache the data
#     image_df.to_pickle('cached_data.pkl')
# # print(image_df['Data'])
# # for index, image_array in enumerate(image_df['Data']):
# #     print(f"Image {index} shape: {image_array.shape}")
# # Reshape image data to (n_samples, n_features)
# # Reshape image data to (n_samples, n_features)
# # Reshape image data to (n_samples, n_features)
# n_samples = len(image_df)
# height, width, _ = image_df['Data'].iloc[0].shape
# n_features = height * width * 3
# X = np.stack(image_df['Data'].apply(lambda x: x.reshape(-1)).tolist())

# # Apply SpectralClustering
# spectral_clustering = SpectralClustering(n_clusters=2)
# cluster_labels = spectral_clustering.fit_predict(X)

# # Perform dimensionality reduction for visualization
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# # Plot the scatter plot
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels)
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.title('Spectral Clustering Results')
# plt.show()
