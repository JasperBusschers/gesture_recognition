import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            24, 2, strides=(2, 2), padding='valid', data_format=None,
            dilation_rate=(1, 1), activation=tf.nn.relu, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
            kernel_constraint=None, bias_constraint=None,input_shape=(32,32,3)
        )
        self.conv2 = tf.keras.layers.Conv2D(
            48, 3, strides=(2, 2), padding='valid', data_format=None,
            dilation_rate=(1, 1), activation=tf.nn.relu, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
            kernel_constraint=None, bias_constraint=None
        )
        self.conv3 = tf.keras.layers.Conv2D(
            96, 2, strides=(2, 2), padding='valid', data_format=None,
            dilation_rate=(1, 1), activation=tf.nn.relu, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
            kernel_constraint=None, bias_constraint=None
        )

    def call(self, input_features):
        activation = self.conv1(input_features)
        activation = self.conv2(activation)
        activation = self.conv3(activation)
        return activation


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = tf.keras.layers.Conv2DTranspose(
            96, 3, strides=(2, 2), padding='valid', output_padding=None,
            data_format=None, activation=tf.nn.relu, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
            kernel_constraint=None, bias_constraint=None, input_shape=(3,3,16)
        )
        self.deconv2 = tf.keras.layers.Conv2DTranspose(
            48, 4, strides=(2, 2), padding='valid', output_padding=None,
            data_format=None, activation=tf.nn.relu, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
            kernel_constraint=None, bias_constraint=None, input_shape=(3, 3, 16)
        )
        self.deconv3 = tf.keras.layers.Conv2DTranspose(
            1, 2, strides=(2, 2), padding='valid', output_padding=None,
            data_format=None, activation=None, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
            kernel_constraint=None, bias_constraint=None, input_shape=(3, 3, 16)
        )

    def call(self, code):
        activation = self.deconv1(code)
        activation = self.deconv2(activation)
        activation = self.deconv3(activation)
        return activation




class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed



def loss(model, original, target):
    reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(original), target)))
    return reconstruction_error



def train( model, opt,original,  target):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model, original, target), model.trainable_variables)
        gradient_variables = zip(gradients, model.trainable_variables)
        opt.apply_gradients(gradient_variables)



