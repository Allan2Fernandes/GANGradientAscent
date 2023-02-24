import tensorflow as tf
import DatasetBuilder
import GAN

dataset_directory_path = "C:/Users/allan/Downloads/GANFacesDateset"
batch_size = 64
target_size = (128,128)
image_shape = (128,128,3)
noise_dimensions = 64
gradient_penalty_weight = 10
learning_rate = 0.0001
epochs = 500

dataset_builder = DatasetBuilder.DatasetBuilder(directory_path=dataset_directory_path, batch_size=batch_size, target_size=target_size)
dataset = dataset_builder.get_dataset()

GAN = GAN.GAN(image_shape=image_shape, noise_dimensions=noise_dimensions)
GAN.initialize_BCE_loss()
GAN.build_generator()
GAN.build_critic()
#GAN.build_discriminator(target_size=image_shape)
GAN.initialize_optimizers(learning_rate=learning_rate)

#GAN.train_model_BCE_loss(epochs=epochs, dataset=dataset)
GAN.train_model_wasserstein(epochs=epochs, dataset=dataset, c_lambda=gradient_penalty_weight)