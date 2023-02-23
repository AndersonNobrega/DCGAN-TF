import tensorflow as tf


class Discriminator:

    def __init__(self, learning_rate):
        self._learning_rate = learning_rate
        self._cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
        self._model = self._create_model()
        self._model.summary()

    def _create_model(self):
        model = tf.keras.Sequential(name='Discriminator')
        model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
        model.add(tf.keras.layers.LeakyReLU(0.2))
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU(0.2))
        model.add(tf.keras.layers.Dropout(0.3))

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
