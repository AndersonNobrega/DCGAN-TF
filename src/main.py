import datetime
import io
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm

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


def get_args():
    parser = ArgumentParser(allow_abbrev=False, description='', formatter_class=ArgumentDefaultsHelpFormatter)

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


def generate_and_save_images(model, test_input, epoch=None, img_path=None):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    if epoch is not None and img_path is not None:
        plt.savefig('{}/image_at_epoch_{:04d}.png'.format(img_path, epoch))

    # Step needed to be compatible with tensorboard
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image


def write_tensorboard_logs(file_writer, label, content, step, content_type='scalar'):
    with file_writer.as_default():
        if content_type == 'scalar':
            tf.summary.scalar(label, content, step=step)
        elif content_type == 'image':
            tf.summary.image(label, content, step=step)


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

    return disc_loss, gen_loss


def train(generator, discriminator, dataset, img_path, epochs, batch_size, num_generate, noise_dim, discriminator_file_writer, generator_file_writer):
    tqdm.write("\n---------- Starting training loop... ----------\n")

    seed = tf.random.normal([num_generate, noise_dim])
    generator_loss_hist = []
    discriminator_loss_hist = []
    step = 0

    for epoch in range(epochs):
        tqdm.write('Epoch: {}/{}'.format(epoch + 1, epochs))

        for batch_index, image_batch in enumerate(tqdm(dataset)):
            disc_loss, gen_loss = train_step(generator, discriminator, image_batch, batch_size, noise_dim)

            discriminator_loss_hist.append(disc_loss)
            generator_loss_hist.append(gen_loss)

            if batch_index % 100 == 0 and batch_index > 0:
                write_tensorboard_logs(discriminator_file_writer, 'Loss', tf.reduce_mean(discriminator_loss_hist), step, 'scalar')
                write_tensorboard_logs(generator_file_writer, 'Loss', tf.reduce_mean(generator_loss_hist), step, 'scalar')

                write_tensorboard_logs(generator_file_writer, 'Generated Images', generate_and_save_images(generator.get_model(), seed), step,
                                       'image')

                step += 1

        # Produce images for the GIF as you go
        generate_and_save_images(generator.get_model(), seed, epoch + 1, img_path)

        tqdm.write(('Discriminator Loss: {:.4f} - Generator Loss: {:.4f}'.format(
            tf.reduce_mean(discriminator_loss_hist), tf.reduce_mean(generator_loss_hist)))
        )

        generator_loss_hist.clear()
        discriminator_loss_hist.clear()

    tqdm.write("\n---------- Training loop finished. ----------\n")


def main():
    # Get CLI args
    args = get_args()

    # Create directory for images
    current_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    img_path = pathlib.Path(__file__).resolve().parents[1] / pathlib.Path("img") / current_time
    img_path.mkdir(parents=True)

    # Tensorboard file writer for training logs
    discriminator_log_dir = "logs/{}/discriminator/".format(current_time)
    generator_log_dir = "logs/{}/generator/".format(current_time)

    discriminator_file_writer = tf.summary.create_file_writer(discriminator_log_dir)
    generator_file_writer = tf.summary.create_file_writer(generator_log_dir)

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
          args['noise_dim'],
          discriminator_file_writer,
          generator_file_writer)

    # Create gif from all images created during training
    create_gif('{}/dcgan.gif'.format(img_path), '{}/image*.png'.format(img_path), delete_file=True)


if __name__ == '__main__':
    main()
