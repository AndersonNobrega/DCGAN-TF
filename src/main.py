import datetime
import os
import pathlib
import time
from argparse import ArgumentParser, RawTextHelpFormatter

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


# checkpoint_dir = '../training_checkpoints/'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator.optimizer(),
#                                  discriminator_optimizer=discriminator.optimizer(),
#                                  generator=generator.get_model(),
#                                  discriminator=discriminator.get_model())

def get_args():
    parser = ArgumentParser(allow_abbrev=False, description='', formatter_class=RawTextHelpFormatter)

    parser.add_argument('-b', '--batch_size', type=int, help='Batch size for the training dataset.', default=256)
    parser.add_argument('-e', '--epochs', type=int, help='Amount of epochs to train model.', default=1)
    parser.add_argument('-l', '--learning_rate', type=float, help='Learning rate for both the generator and discriminator models.', default=3e-4)
    parser.add_argument('-n', '--noise_dim', type=int, help='Dimension for noise vector used by the generator.', default=100)
    parser.add_argument('-u', '--num_generate', type=int, help='Dimension for noise vector used by the generator.', default=16)
    parser.add_argument('-s', '--buffer_size', type=int, help='Buffer size for dataset shuffle.', default=60000)

    return vars(parser.parse_args())


def load_dataset(buffer_size, batch_size):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    # Batch and shuffle the data
    return tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)


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
def train_step(generator, discriminator, images, batch_size, noise_dim):
    noise = tf.random.normal([batch_size, noise_dim])

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


def train(generator, discriminator, dataset, img_path, epochs, batch_size, num_generate, noise_dim):
    print("\n---------- Starting training loop... ----------\n")

    seed = tf.random.normal([num_generate, noise_dim])

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(generator, discriminator, image_batch, batch_size, noise_dim)

        # Produce images for the GIF as you go
        generate_and_save_images(generator.get_model(), epoch + 1, seed, img_path)

        # Save the model every 15 epochs
        # if (epoch + 1) % 15 == 0:
        #     checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch: {}/{} - Time: {:.2f} seconds'.format(epoch + 1, epochs, time.time() - start))

    print("\n---------- Training loop finished. ----------\n")


def main():
    # Get CLI args
    args = get_args()

    # Create directory for images
    img_path = pathlib.Path(__file__).resolve().parents[1] / pathlib.Path("img") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    img_path.mkdir(parents=True)

    # Create Generator and Discriminator models
    generator = Generator(learning_rate=args['learning_rate'])
    discriminator = Discriminator(learning_rate=args['learning_rate'])

    # Train loop
    train(generator,
          discriminator,
          load_dataset(args['buffer_size'],
                       args['batch_size']),
          img_path,
          args['epochs'],
          args['batch_size'],
          args['num_generate'],
          args['noise_dim'])

    # Create gif from all images created during training
    create_gif('{}/dcgan.gif'.format(img_path), '{}/image*.png'.format(img_path), delete_file=True)


if __name__ == '__main__':
    main()
