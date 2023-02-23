import tensorflow as tf


class Generator:

    def __init__(self, learning_rate):
        self._FEATURE_MAP = 64
        self._learning_rate = learning_rate
        self._cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
        self._model = self._create_model()
        self._model.summary()

    def _create_model(self):
        model = tf.keras.Sequential(name='Generator')
        model.add(tf.keras.layers.Dense(7 * 7 * (self._FEATURE_MAP * 4), use_bias=False, input_shape=(100,)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Reshape((7, 7, (self._FEATURE_MAP * 4))))

        model.add(tf.keras.layers.Conv2DTranspose(self._FEATURE_MAP * 4, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False,
                  kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Conv2DTranspose(self._FEATURE_MAP * 2, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False,
                  kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Conv2DTranspose(1, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh',
                  kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)))

        return model

    def get_model(self):
        return self._model

    def loss(self, fake_output):
        return self._cross_entropy(tf.ones_like(fake_output), fake_output)

    def optimizer(self):
        return self._optimizer
