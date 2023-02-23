import datetime
import os
import pathlib
import time

import matplotlib.pyplot as plt

# Remove Tensorflow log spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Check for available GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print('Using GPU for model training.\n')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
else:
    print('No GPU available for model training. Using CPU instead.\n')

from model import Generator, Discriminator
from utils import create_gif

BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 25
NOISE_DIM = 100
NUM_GENERATE = 16
SEED = tf.random.normal([NUM_GENERATE, NOISE_DIM])
LEARNING_RATE = 2e-4


# checkpoint_dir = '../training_checkpoints/'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator.optimizer(),
#                                  discriminator_optimizer=discriminator.optimizer(),
#                                  generator=generator.get_model(),
#                                  discriminator=discriminator.get_model())

def load_dataset():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    # Batch and shuffle the data
    return tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def generate_and_save_images(model, epoch, test_input, img_path):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('{}/image_at_epoch_{:04d}.png'.format(img_path, epoch))
    plt.close(fig)


@tf.function
def train_step(generator, discriminator, images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator.get_model()(noise, training=True)

        real_output = discriminator.get_model()(images, training=True)
        fake_output = discriminator.get_model()(generated_images, training=True)

        gen_loss = generator.loss(fake_output)
        disc_loss = discriminator.loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.get_model().trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.get_model().trainable_variables)

    generator.optimizer().apply_gradients(zip(gradients_of_generator, generator.get_model().trainable_variables))
    discriminator.optimizer().apply_gradients(zip(gradients_of_discriminator, discriminator.get_model().trainable_variables))


def train(generator, discriminator, dataset, img_path):
    print("\n---------- Starting training loop... ----------\n")

    for epoch in range(EPOCHS):
        start = time.time()

        for image_batch in dataset:
            train_step(generator, discriminator, image_batch)

        # Produce images for the GIF as you go
        generate_and_save_images(generator.get_model(), epoch + 1, SEED, img_path)

        # Save the model every 15 epochs
        # if (epoch + 1) % 15 == 0:
        #     checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch: {}/{} - Time: {:.2f} seconds'.format(epoch + 1, EPOCHS, time.time() - start))

    print("\n---------- Training loop finished. ----------\n")


def main():
    # Create directory for images
    img_path = pathlib.Path(__file__).resolve().parents[1] / pathlib.Path("img") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    img_path.mkdir(parents=True)

    # Create Generator and Discriminator models
    generator = Generator(learning_rate=LEARNING_RATE)
    discriminator = Discriminator(learning_rate=LEARNING_RATE)

    # Train loop
    train(generator, discriminator, load_dataset(), img_path)

    # Create gif from all images created during training
    create_gif('{}/dcgan.gif'.format(img_path), '{}/image*.png'.format(img_path), delete_file=True)


if __name__ == '__main__':
    main()
