import tensorflow as tf


class Generator:

    def __init__(self, learning_rate):
        self._learning_rate = learning_rate
        self._cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
        self._model = self._create_model()
        self._model.summary()

    def _create_model(self):
        model = tf.keras.Sequential(name='Generator')
        model.add(tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        return model

    def get_model(self):
        return self._model

    def loss(self, fake_output):
        return self._cross_entropy(tf.ones_like(fake_output), fake_output)

    def optimizer(self):
        return self._optimizer
