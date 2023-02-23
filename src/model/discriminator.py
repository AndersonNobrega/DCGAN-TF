import tensorflow as tf


class Discriminator:

    def __init__(self, learning_rate):
        self._FEATURE_MAP = 64
        self._learning_rate = learning_rate
        self._cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
        self._model = self._create_model()
        self._model.summary()

    def _create_model(self):
        model = tf.keras.Sequential(name='Discriminator')
        model.add(tf.keras.layers.Conv2D(self._FEATURE_MAP * 2, input_shape=[28, 28, 1], kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False,
                                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)))
        model.add(tf.keras.layers.LeakyReLU(0.2))

        model.add(tf.keras.layers.Conv2D(self._FEATURE_MAP * 4, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False,
                                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(0.2))

        model.add(tf.keras.layers.Conv2D(self._FEATURE_MAP * 8, kernel_size=(5, 5), strides=(1, 1), use_bias=False,
                                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.02), activation='sigmoid'))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1))

        return model

    def get_model(self):
        return self._model

    def loss(self, real_output, fake_output):
        real_loss = self._cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self._cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def optimizer(self):
        return self._optimizer
