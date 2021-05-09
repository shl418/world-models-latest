import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

INPUT_DIM = (64,64,3)

CONV_FILTERS = [32,64,64, 128]
CONV_FILTERS_DIS = [16,64,128,128]
CONV_KERNEL_SIZES = [4,4,4,4]
CONV_STRIDES = [2,2,2,2]
CONV_ACTIVATIONS = ['relu','relu','relu','relu']

DENSE_SIZE = 1024

CONV_T_FILTERS = [64,64,32,3]
CONV_T_KERNEL_SIZES = [5,5,6,6]
CONV_T_STRIDES = [2,2,2,2]
CONV_T_ACTIVATIONS = ['relu','relu','relu','sigmoid']

Z_DIM = 32

BATCH_SIZE = 100
LEARNING_RATE = 0.0001
KL_TOLERANCE = 0.5




class Sampling(Layer):
    def call(self, inputs):
        mu, log_var = inputs
        epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
        return mu + K.exp(log_var / 2) * epsilon



class VAEGANModel(Model):
    def __init__(self, encoder, decoder, distriminator, r_loss_factor, lr, **kwargs):
        super(VAEGANModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = distriminator
        self.lr = lr

        self.enc_optimizer = tf.keras.optimizers.Adam(self.lr)
        self.dec_optimizer = tf.keras.optimizers.Adam(self.lr)
        self.disc_optimizer = tf.keras.optimizers.Adam(self.lr)
        
        self.r_loss_factor = r_loss_factor


    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        lattent_r =  tf.random.normal((BATCH_SIZE, Z_DIM))
        with tf.GradientTape(persistent=True) as tape:
            z_mean, z_log_var, z = self.encoder(data)
            # compute kl_loss
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_sum(kl_loss, axis = 1)
            kl_loss *= -0.5
            print(kl_loss)
            fake = self.decoder(z)
            # print(fake.shape)
            dis_fake,dis_inner_fake = self.discriminator(fake)
            # print(dis_fake.shape,dis_inner_fake.shape)
            dis_fake_r,_=self.discriminator(self.decoder(lattent_r))
            dis_true,dis_inner_true = self.discriminator(data)

            vae_inner = dis_inner_fake-dis_inner_true
            vae_inner = vae_inner*vae_inner

            mean,var = tf.nn.moments(self.encoder(data)[0],axes=0)
            var_to_one = var - 1

            normal_loss = tf.reduce_mean(mean*mean) + tf.reduce_mean(var_to_one*var_to_one)
        
            kl_loss = tf.reduce_mean(kl_loss)
            vae_diff_loss = tf.reduce_mean(vae_inner)
            f_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(dis_fake), dis_fake))
            r_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(dis_fake_r), dis_fake_r))
            t_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(dis_true), dis_true))
            gan_loss = (0.5*t_dis_loss + 0.25*f_dis_loss + 0.25*r_dis_loss)
            vae_loss = tf.reduce_mean(tf.abs(data-fake)) 
            E_loss = vae_diff_loss + self.r_loss_factor*kl_loss + 0.1*normal_loss
            G_loss = vae_diff_loss - gan_loss
            D_loss = gan_loss
        E_grad = tape.gradient(E_loss,self.encoder.trainable_variables)
        G_grad = tape.gradient(G_loss,self.decoder.trainable_variables)
        D_grad = tape.gradient(D_loss,self.discriminator.trainable_variables)
        del tape
        self.enc_optimizer.apply_gradients(zip(E_grad, self.encoder.trainable_variables))
        self.dec_optimizer.apply_gradients(zip(G_grad, self.decoder.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(D_grad, self.discriminator.trainable_variables))

        return [gan_loss, vae_loss, f_dis_loss, r_dis_loss, t_dis_loss, vae_diff_loss, E_loss, D_loss, kl_loss, normal_loss]
    
    def call(self,inputs):
        latent = self.encoder(inputs)
        return self.decoder(latent)



class VAE():
    def __init__(self):
        self.models = self._build()
        self.full_model = self.models[0]
        self.encoder = self.models[1]
        self.decoder = self.models[2]
        self.discriminator = self.models[3]

        self.input_dim = INPUT_DIM
        self.z_dim = Z_DIM
        self.learning_rate = LEARNING_RATE
        self.kl_tolerance = KL_TOLERANCE

        self.metrics_names = ["gan_loss", "vae_loss", "fake_dis_loss", "r_dis_loss", "t_dis_loss", "vae_inner_loss", "E_loss", "D_loss", "kl_loss", "normal_loss"]
        self.metrics = []
        for m in self.metrics_names :
            self.metrics.append(tf.keras.metrics.Mean('m', dtype=tf.float32))

    def _build(self):
        vae_x = Input(shape=INPUT_DIM, name='observation_input')
        vae_c1 = Conv2D(filters = CONV_FILTERS[0], kernel_size = CONV_KERNEL_SIZES[0], strides = CONV_STRIDES[0], activation=CONV_ACTIVATIONS[0], name='conv_layer_1')(vae_x)
        vae_c2 = Conv2D(filters = CONV_FILTERS[1], kernel_size = CONV_KERNEL_SIZES[1], strides = CONV_STRIDES[1], activation=CONV_ACTIVATIONS[0], name='conv_layer_2')(vae_c1)
        vae_c3= Conv2D(filters = CONV_FILTERS[2], kernel_size = CONV_KERNEL_SIZES[2], strides = CONV_STRIDES[2], activation=CONV_ACTIVATIONS[0], name='conv_layer_3')(vae_c2)
        vae_c4= Conv2D(filters = CONV_FILTERS[3], kernel_size = CONV_KERNEL_SIZES[3], strides = CONV_STRIDES[3], activation=CONV_ACTIVATIONS[0], name='conv_layer_4')(vae_c3)

        vae_z_in = Flatten()(vae_c4)

        vae_z_mean = Dense(Z_DIM, name='mu')(vae_z_in)
        vae_z_log_var = Dense(Z_DIM, name='log_var')(vae_z_in)

        vae_z = Sampling(name='z')([vae_z_mean, vae_z_log_var])
        

        #### DECODER: 
        vae_z_input = Input(shape=(Z_DIM,), name='z_input')

        vae_dense = Dense(1024, name='dense_layer')(vae_z_input)
        vae_unflatten = Reshape((1,1,DENSE_SIZE), name='unflatten')(vae_dense)
        vae_d1 = Conv2DTranspose(filters = CONV_T_FILTERS[0], kernel_size = CONV_T_KERNEL_SIZES[0] , strides = CONV_T_STRIDES[0], activation=CONV_T_ACTIVATIONS[0], name='deconv_layer_1')(vae_unflatten)
        vae_d2 = Conv2DTranspose(filters = CONV_T_FILTERS[1], kernel_size = CONV_T_KERNEL_SIZES[1] , strides = CONV_T_STRIDES[1], activation=CONV_T_ACTIVATIONS[1], name='deconv_layer_2')(vae_d1)
        vae_d3 = Conv2DTranspose(filters = CONV_T_FILTERS[2], kernel_size = CONV_T_KERNEL_SIZES[2] , strides = CONV_T_STRIDES[2], activation=CONV_T_ACTIVATIONS[2], name='deconv_layer_3')(vae_d2)
        vae_d4 = Conv2DTranspose(filters = CONV_T_FILTERS[3], kernel_size = CONV_T_KERNEL_SIZES[3] , strides = CONV_T_STRIDES[3], activation=CONV_T_ACTIVATIONS[3], name='deconv_layer_4')(vae_d3)
        
        #### DISCRIMINATOR:

        vae_zp_input = Input(shape=INPUT_DIM,name="dis_input")

        vae_s1 = Conv2D(filters = CONV_FILTERS_DIS[0], kernel_size = CONV_KERNEL_SIZES[0], strides = CONV_STRIDES[0], activation=CONV_ACTIVATIONS[0], name='dis_conv_layer_1')(vae_zp_input)
        vae_s2 = Conv2D(filters = CONV_FILTERS_DIS[1], kernel_size = CONV_KERNEL_SIZES[1], strides = CONV_STRIDES[1], activation=CONV_ACTIVATIONS[0], name='dis_conv_layer_2')(vae_s1)
        vae_s3= Conv2D(filters = CONV_FILTERS_DIS[2], kernel_size = CONV_KERNEL_SIZES[2], strides = CONV_STRIDES[2], activation=CONV_ACTIVATIONS[0], name='dis_conv_layer_3')(vae_s2)
        vae_s4= Conv2D(filters = CONV_FILTERS_DIS[3], kernel_size = CONV_KERNEL_SIZES[3], strides = CONV_STRIDES[3], activation=CONV_ACTIVATIONS[0], name='dis_conv_layer_4')(vae_s3)
        inner = Flatten()(vae_s4)
        vae_s_in = Flatten()(vae_s4)

        vae_dis_flat = Dense(CONV_FILTERS_DIS[3], name='dis_flat1')(vae_s_in)
        vae_dis_flat2 = Dense(1, name='dis_flat2')(vae_dis_flat)

        #### MODELS

    
        vae_encoder = Model(vae_x, [vae_z_mean, vae_z_log_var, vae_z], name = 'encoder')
        vae_decoder = Model(vae_z_input, vae_d4, name = 'decoder')
        vae_discriminator = Model(vae_zp_input, [vae_dis_flat2,inner],name='discriminator')

        vae_full = VAEGANModel(vae_encoder, vae_decoder, vae_discriminator, LEARNING_RATE, 100)
        vae_full.compile('Adam')

        return (vae_full,vae_encoder, vae_decoder, vae_discriminator)

    def set_weights(self, filepath):
        self.full_model.load_weights(filepath)

    def train(self, data):
        for i in (10,len(data)):
            if i % 10 == 0:
                self.print_metrics(i)
            results = self.full_model.train_step(data[i-10:i])
            for metric,result in zip(self.metrics, results) :
                metric(result)
        
    def print_metrics(self,step) :
        s = ""
        for name,metric in zip(self.metrics_names,self.metrics) :
            s+= " " + name + " " + str(np.around(metric.result().numpy(), 3)) 
        print(f"\rStep : " + str(step) + " " + s, end="", flush=True)
        for metric in self.metrics : 
            metric.reset_states()

    def save_weights(self, filepath):
        self.full_model.save_weights(filepath)
