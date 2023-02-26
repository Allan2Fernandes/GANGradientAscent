import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, Reshape, Dense, LeakyReLU, Flatten, Conv2DTranspose, Lambda, UpSampling2D, Activation
from keras.activations import selu
import matplotlib.pyplot as plt

import time


class GAN:
    def __init__(self, image_shape, noise_dimensions):
        self.image_shape = image_shape
        self.image_dim = self.image_shape[0]
        self.noise_dimensions = noise_dimensions
        pass

    def build_generator(self):
        filters = 64
        kernel_size = (3, 3)
        padding = 'same'
        kernel_initializer = 'he_normal'

        self.generator = tf.keras.models.Sequential([
            Dense(units=(self.image_dim * self.image_dim * 3*2*2), input_shape=[self.noise_dimensions]),
            Reshape(target_shape=(self.image_dim*2, self.image_dim*2, 3)),
            # Downsample it to the bottleneck
            Conv2D(filters=filters, kernel_size=kernel_size, strides=2, padding=padding, use_bias=False,kernel_initializer=kernel_initializer),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(filters=filters * 2, kernel_size=kernel_size, strides=2, padding=padding, use_bias=False,kernel_initializer=kernel_initializer),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(filters=filters * 4, kernel_size=kernel_size, strides=2, padding=padding, use_bias=False,kernel_initializer=kernel_initializer),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(filters=filters * 8, kernel_size=kernel_size, strides=2, padding=padding, use_bias=False,kernel_initializer=kernel_initializer),
            BatchNormalization(),
            LeakyReLU(),

            # Bottleneck layer
            Conv2D(filters=filters * 16, kernel_size=kernel_size, strides=2, padding=padding,kernel_initializer='he_normal', use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            # Upsample it to the input shape
            Conv2DTranspose(filters=filters * 4, strides=2, kernel_size=kernel_size, padding=padding,use_bias=False),
            BatchNormalization(),
            tf.keras.layers.Activation('selu'),
            Conv2DTranspose(filters=filters * 4, strides=1, kernel_size=kernel_size, padding=padding, use_bias=False),
            BatchNormalization(),
            tf.keras.layers.Activation('selu'),
            Conv2DTranspose(filters=filters * 2, strides=2, kernel_size=kernel_size, padding=padding,use_bias=False),
            BatchNormalization(),
            tf.keras.layers.Activation('selu'),
            Conv2DTranspose(filters=filters * 2, strides=1, kernel_size=kernel_size, padding=padding, use_bias=False),
            BatchNormalization(),
            tf.keras.layers.Activation('selu'),
            Conv2DTranspose(filters=filters, strides=2, kernel_size=kernel_size, padding=padding,use_bias=False),
            BatchNormalization(),
            tf.keras.layers.Activation('selu'),
            Conv2DTranspose(filters=filters, strides=1, kernel_size=kernel_size, padding=padding, use_bias=False),
            BatchNormalization(),
            tf.keras.layers.Activation('selu'),
            Conv2DTranspose(filters=3, activation='tanh', strides=2, kernel_size=kernel_size,padding=padding, use_bias=False)
        ])

        # scaled_image_dim = self.image_dim // 64 #Needs to be scaled up 4 times
        #
        #
        # self.generator = tf.keras.models.Sequential([
        #     Dense(units=(scaled_image_dim * scaled_image_dim * filters), input_shape=[self.noise_dimensions]),
        #     BatchNormalization(),
        #     LeakyReLU(),
        #     Reshape(target_shape=(scaled_image_dim, scaled_image_dim, filters)),
        #     Conv2DTranspose(filters=filters//2, strides=1, kernel_size=kernel_size, padding=padding,use_bias=False),
        #     BatchNormalization(),
        #     tf.keras.layers.Activation('selu'),
        #     Conv2DTranspose(filters=filters // 2, strides=2, kernel_size=kernel_size, padding=padding, use_bias=False),
        #     BatchNormalization(),
        #     tf.keras.layers.Activation('selu'),
        #     Conv2DTranspose(filters=filters//4,  strides=1, kernel_size=kernel_size,padding=padding, use_bias=False),
        #     BatchNormalization(),
        #     tf.keras.layers.Activation('selu'),
        #     Conv2DTranspose(filters=filters // 4, strides=2, kernel_size=kernel_size, padding=padding, use_bias=False),
        #     BatchNormalization(),
        #     tf.keras.layers.Activation('selu'),
        #     Conv2DTranspose(filters=filters//8, strides=1, kernel_size=kernel_size,padding=padding, use_bias=False),
        #     BatchNormalization(),
        #     tf.keras.layers.Activation('selu'),
        #     Conv2DTranspose(filters=filters // 8, strides=2, kernel_size=kernel_size, padding=padding, use_bias=False),
        #     BatchNormalization(),
        #     tf.keras.layers.Activation('selu'),
        #     Conv2DTranspose(filters=filters//16, strides=1, kernel_size=kernel_size,padding=padding, use_bias=False),
        #     BatchNormalization(),
        #     tf.keras.layers.Activation('selu'),
        #     Conv2DTranspose(filters=filters // 16, strides=2, kernel_size=kernel_size, padding=padding, use_bias=False),
        #     BatchNormalization(),
        #     tf.keras.layers.Activation('selu'),
        #     Conv2DTranspose(filters=filters // 32, strides=1, kernel_size=kernel_size,padding=padding, use_bias=False),
        #     BatchNormalization(),
        #     tf.keras.layers.Activation('selu'),
        #     Conv2DTranspose(filters=filters // 32, strides=2, kernel_size=kernel_size, padding=padding, use_bias=False),
        #     BatchNormalization(),
        #     tf.keras.layers.Activation('selu'),
        #     Conv2DTranspose(filters=filters // 64,strides=1, kernel_size=kernel_size,padding=padding, use_bias=False),
        #     BatchNormalization(),
        #     tf.keras.layers.Activation('selu'),
        #     Conv2DTranspose(filters=filters // 64, strides=2, kernel_size=kernel_size, padding=padding, use_bias=False),
        #     BatchNormalization(),
        #     tf.keras.layers.Activation('selu'),
        #     Conv2DTranspose(filters = 3, activation='tanh', strides=1, kernel_size=kernel_size, padding=padding, use_bias=False)
        # ])

        self.generator.summary()
        pass

    def build_generatorv2(self):
        filters = 2048
        kernel_size = (3, 3)
        padding = 'same'
        kernel_initializer = 'he_normal'
        scaled_image_dim = self.image_dim // 64  # Needs to be scaled up 4 times



        self.generator = tf.keras.models.Sequential([
            Dense(units=(scaled_image_dim * scaled_image_dim * filters), input_shape=[self.noise_dimensions]),
            BatchNormalization(),
            LeakyReLU(),
            Reshape(target_shape=(scaled_image_dim, scaled_image_dim, filters)),
            UpSampling2D(size=(2, 2), interpolation='nearest'),
            Conv2D(filters=filters // 2, strides=1, kernel_size=kernel_size, padding=padding, use_bias=False),
            BatchNormalization(),
            Lambda(lambda x: selu(x)),
            UpSampling2D(size=(2, 2), interpolation='nearest'),
            Conv2D(filters=filters // 4, strides=1, kernel_size=kernel_size, padding=padding, use_bias=False),
            BatchNormalization(),
            Lambda(lambda x: selu(x)),
            UpSampling2D(size=(2, 2), interpolation='nearest'),
            Conv2D(filters=filters // 8, strides=1, kernel_size=kernel_size, padding=padding, use_bias=False),
            BatchNormalization(),
            Lambda(lambda x: selu(x)),
            UpSampling2D(size=(2, 2), interpolation='nearest'),
            Conv2D(filters=filters // 16, strides=1, kernel_size=kernel_size, padding=padding, use_bias=False),
            BatchNormalization(),
            Lambda(lambda x: selu(x)),
            UpSampling2D(size=(2, 2), interpolation='nearest'),
            Conv2D(filters=filters // 32, strides=1, kernel_size=kernel_size, padding=padding, use_bias=False),
            BatchNormalization(),
            Lambda(lambda x: selu(x)),
            UpSampling2D(size=(2, 2), interpolation='nearest'),
            Conv2D(filters=filters // 64, strides=1, kernel_size=kernel_size, padding=padding, use_bias=False),
            BatchNormalization(),
            Lambda(lambda x: selu(x)),
            Conv2D(filters=3, activation='tanh', strides=1, kernel_size=kernel_size, padding=padding, use_bias=False)
        ])
        self.generator.summary()
        pass

    def build_critic(self):
        filters = 64
        kernel_size = (4, 4)
        padding = 'same'
        kernel_initializer = 'he_normal'

        self.critic = tf.keras.models.Sequential([
            Input(shape=self.image_shape),

            Conv2D(filters=filters, strides=(2, 2), kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer,use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(filters=filters, strides=(2, 2), kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer,use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(filters=filters * 2, strides=(2, 2), kernel_size=kernel_size, padding=padding,kernel_initializer=kernel_initializer,use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(filters=filters * 2, strides=(2, 2), kernel_size=kernel_size, padding=padding,kernel_initializer=kernel_initializer,use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(filters=filters * 4, strides=(2, 2), kernel_size=kernel_size, padding=padding,kernel_initializer=kernel_initializer,use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(filters=filters * 4, strides=(2, 2), kernel_size=kernel_size, padding=padding,kernel_initializer=kernel_initializer,use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Flatten(),
            Dense(units=1)
        ])
        self.critic.summary()
        pass




    def build_discriminator(self, target_size):
        filters = 64
        kernel_size = (4, 4)
        padding = 'same'
        kernel_initializer = 'he_normal'

        self.discriminator = tf.keras.models.Sequential([
            Input(shape=self.image_shape),

            Conv2D(filters=filters, strides=(2, 2), kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer,use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(filters=filters, strides=(2, 2), kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer,use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(filters=filters * 2, strides=(2, 2), kernel_size=kernel_size, padding=padding,kernel_initializer=kernel_initializer,use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(filters=filters * 2, strides=(2, 2), kernel_size=kernel_size, padding=padding,kernel_initializer=kernel_initializer,use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(filters=filters * 4, strides=(2, 2), kernel_size=kernel_size, padding=padding,kernel_initializer=kernel_initializer,use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(filters=filters * 4, strides=(2, 2), kernel_size=kernel_size, padding=padding,kernel_initializer=kernel_initializer,use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Flatten(),
            Dense(units=1, activation='sigmoid')
        ])

        # self.prebuilt_model = MobileNetV3Large(
        #     input_shape=target_size,
        #     alpha=1.0,
        #     minimalistic=False,
        #     include_top=False,
        #     weights=None,
        #     input_tensor=None,
        #     classes=2,
        #     pooling=None,
        #     dropout_rate=0.2,
        #     classifier_activation="sigmoid",
        #     include_preprocessing=False,
        # )
        # self.prebuilt_model.trainable = True
        # input_layer = self.prebuilt_model.input
        # flatten_layer = Flatten()(self.prebuilt_model.output)
        # classification_layer = Dense(units=1, activation='sigmoid')(flatten_layer)
        # self.discriminator = tf.keras.Model(inputs=input_layer, outputs=classification_layer)

        self.discriminator.summary()
        pass

    def initialize_optimizers(self, learning_rate):
        self.gen_optimizer = tf.keras.optimizers.RMSprop(learning_rate=.0001, clipvalue=1.0)#, decay=1e-8)
        self.critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=.0001, clipvalue=1.0)#, decay=1e-8)
        # self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=1.0)
        # self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=1.0)
        pass

    def get_gradient(self, critic, real_images, fake_images, weight):
        mixed_images = real_images*weight + fake_images*(1*weight)
        with tf.GradientTape() as tape:
            tape.watch(mixed_images)
            mixed_scores = critic(mixed_images)
            pass
        gradient = tape.gradient(mixed_scores, mixed_images)
        return gradient

    def gradient_penalty(self, gradient, batch_size):
        gradient = tf.reshape(gradient, [batch_size, -1])
        gradient_norm = tf.norm(gradient)
        penalty = tf.reduce_mean((gradient_norm-1)**2)
        return penalty

    def get_crit_loss(self, crit_fake_score, crit_real_score, gradient_penalty, c_lambda):
        total_crit_loss = -crit_real_score + crit_fake_score + c_lambda * gradient_penalty
        return total_crit_loss

    def get_gen_loss(self, crit_fake_pred):
        gen_loss = -tf.reduce_mean(crit_fake_pred)
        return gen_loss

    def generate_image(self):
        noise = tf.random.normal(shape=[1, self.noise_dimensions])
        fake_image = self.generator(noise)
        fake_image = fake_image*0.5 + 0.5
        plt.imshow(fake_image[0])
        plt.show()
        pass

    def initialize_BCE_loss(self):
        self.loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
        pass

    def train_model_BCE_loss(self, epochs, dataset):
        for epoch in range(1, epochs+1):
            for step, real_images in enumerate(dataset):
                # Calculate total num of steps in the epoch
                num_steps = len(dataset)
                with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
                    #Get the batch size
                    batch_size = real_images.shape[0]
                    #Create noise
                    noise = tf.random.normal([batch_size, self.noise_dimensions])
                    #Generate fake images
                    fake_images = self.generator(noise)
                    # Get disc predictions on fake images
                    fake_predictions = self.discriminator(fake_images)
                    # Get the predictions on real images
                    real_predictions = self.discriminator(real_images)

                    #Create labels for real and fake
                    real_labels = tf.ones_like(real_predictions)
                    fake_labels = tf.zeros_like(fake_predictions)
                    #Calculate total discriminator loss
                    disc_real_loss = self.loss_function(y_true = real_labels, y_pred = real_predictions)
                    disc_fake_loss = self.loss_function(y_true = fake_labels, y_pred = fake_predictions)
                    total_disc_loss = tf.concat([disc_real_loss, disc_fake_loss], axis = 0)

                    gen_loss = self.loss_function(y_true = real_labels, y_pred = fake_predictions)
                    pass
                #Train the discriminator
                #Diff. disc loss wrt. to disc variables
                disc_gradient = disc_tape.gradient(total_disc_loss, self.discriminator.trainable_variables)
                #Gradient descent
                self.critic_optimizer.apply_gradients(zip(disc_gradient, self.discriminator.trainable_variables))

                #Train the generator
                #Diff. gen loss wrt. to gen variables
                gen_gradient = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                #Gradient descent
                self.gen_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
                print("Epoch: {0} Step: {1}/{4} || Discriminator loss = {2} || Generator loss = {3}".format(epoch,(step + 1),tf.reduce_sum(total_disc_loss),tf.reduce_sum(gen_loss),num_steps))
                pass #end of step
            self.generate_image()
            pass #end of epoch
        pass #end of method

    def train_model_wasserstein(self, epochs, dataset, c_lambda):
        for epoch in range(1, epochs+1):
            epoch_start_time = time.time()
            for step, real_images in enumerate(dataset):
                with tf.GradientTape() as critic_tape, tf.GradientTape() as generator_tape:
                    # Get the batch size
                    batch_size = real_images.shape[0]
                    num_steps = len(dataset)
                    # Create noise
                    noise = tf.random.normal([batch_size, self.noise_dimensions], 0.0, 1.0)
                    # Generate fake images from the noise
                    fake_images = self.generator(noise)

                    # Get the gradient penalty
                    # Get a braodcastable weight
                    weight = tf.random.normal(shape=[batch_size, 1, 1, 1])
                    #Get the score's gradient with respect to the mixed images
                    disc_score_gradient = self.get_gradient(critic=self.critic, real_images=real_images,fake_images=fake_images, weight=weight)
                    #Calculate the gradient penalty
                    gradient_penalty = self.gradient_penalty(gradient=disc_score_gradient, batch_size=batch_size)

                    #Get critic scores
                    #Get the fake image scores
                    fake_scores = self.critic(fake_images)
                    #Get the real scores
                    real_scores = self.critic(real_images)

                    #Use the real and fake image scores to get the total critic loss
                    total_crit_loss = self.get_crit_loss(crit_fake_score=fake_scores, crit_real_score=real_scores, gradient_penalty=gradient_penalty, c_lambda=c_lambda)

                    #Calculate the total gen loss
                    total_gen_loss = self.get_gen_loss(crit_fake_pred=fake_scores)
                    pass

                # Train the critic
                # Gradient of crit loss
                critic_grad = critic_tape.gradient(total_crit_loss, self.critic.trainable_variables)
                # Gradient descent
                self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

                # Train the generator
                # Gradient of gen loss
                gen_grad = generator_tape.gradient(total_gen_loss, self.generator.trainable_variables)
                #Gradient descent
                self.gen_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))
                est_step_time = (time.time() - epoch_start_time)/(step+1)
                print("Epoch: {0} Step: {1}/{4} || Discriminator loss = {2:.8f} || Generator loss = {3:.8f} || Time/Step = {5:.4f}s || Est. Time/Epoch ={6:.4f}s".format(epoch, (step+1),tf.reduce_sum(total_crit_loss),tf.reduce_sum(total_gen_loss), num_steps, est_step_time, (est_step_time*num_steps)))
                pass #End of step
            if epoch%1 == 0:
                self.generate_image()
                tf.keras.models.save_model(self.generator, f"AutoencoderKS3/Epoch{epoch}/Generator.h5")
                tf.keras.models.save_model(self.critic, f"AutoencoderKS3/Epoch{epoch}/Critic.h5")
            pass #End of epoch
        pass #End of method




