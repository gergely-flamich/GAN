import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv2D, Flatten, Dropout, BatchNormalization

import matplotlib.pyplot as plt
import numpy as np

import argparse

from utils import is_valid_file

models = {
    "baseline": None
}

def mnist_input_fn(data, num_epochs=50, batch_size=128, shuffle_samples=5000):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(shuffle_samples)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(mnist_parse_fn)
    dataset = dataset.batch(batch_size)

    return dataset


def mnist_parse_fn(data):
    return (tf.cast(data, tf.float32) - 127.5)/ 127.5



def discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(28 * 28, )))
    model.add(Dense(400))
    model.add(Dropout(0.5))
    model.add(Dense(400))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    return model


def generator():
    model = Sequential()

    model.add(Flatten(input_shape=(100,)))
    model.add(Dense(200))
    model.add(BatchNormalization())
    model.add(Dense(400))
    model.add(BatchNormalization())
    model.add(Dense(28 * 28))
    model.add(BatchNormalization())
    model.add(Reshape((28, 28)))

    return model

def run(args):

    ((train_data, _),
     (eval_data, _)) = tf.keras.datasets.mnist.load_data()

    dataset = mnist_input_fn(train_data)

    gen = generator()
    disc = discriminator()

    disc_losses = []
    gen_losses = []

    cross_entropy = tf.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.)

    # We'll use Adam to optimize
    trainer = tf.optimizers.Adam(learning_rate=1e-3)

    # Training
    for real_data in dataset:

        # Sample z ~ p(z)
        noise = tf.random.uniform(shape=[tf.shape(real_data)[0], 100],
                                  minval=0.,
                                  maxval=1.)

        # Record the gradients of the following transformations for backprop
        # The gradient tape must be persistent, because we need the gradieants once
        # for the discriminator and once for the generator
        with tf.GradientTape(persistent=True) as tape:

            # Transform sample: x = G(z)
            fake_data = gen.apply(noise)

            # Get the discriminator's opinion
            D_x = disc.apply(real_data)
            D_G_z = disc.apply(fake_data)

            data_shape = tf.shape(D_x)

            # Discriminator loss:
            # E_p_data(x) [log D(x)] + E_p_gen(x) [log 1 - D(G(x))]
            discriminator_loss = cross_entropy(tf.ones_like(D_x), D_x) + cross_entropy(tf.zeros_like(D_G_z), D_G_z)

            # print(tf.math.log(D_x))
            # print(tf.math.log(1 - D_G_z))
            # print(discriminator_loss)
            # break

            # Generator loss:
            # E_p_gen(x) [log D(G(x))]
            generator_loss = cross_entropy(tf.ones_like(D_G_z), D_G_z)

        # Update models simultaneously
        disc_grads = tape.gradient(discriminator_loss, disc.trainable_variables)
        gen_grads = tape.gradient(generator_loss, gen.trainable_variables)

        # Delete the tape now that we have all gradients from it
        del tape

        trainer.apply_gradients(zip(disc_grads, disc.trainable_variables))
        trainer.apply_gradients(zip(gen_grads, gen.trainable_variables))

        # Log some stuff
        disc_losses.append(discriminator_loss)
        gen_losses.append(generator_loss)

        out_format = "Discriminator Loss: {:.2f}, \t Generator Loss: {:.2f}"
        print(out_format.format(discriminator_loss, generator_loss))



    plt.plot(disc_losses)
    plt.plot(gen_losses)

    plt.figure()
    noise = tf.random.uniform(shape=[1, 100],
                              minval=0.,
                              maxval=1.)
    plt.imshow(gen.apply(noise)[0, :, :])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bayes By Backprop models')

    parser.add_argument('--model', choices=list(models.keys()), default='baseline',
                    help='The model to train.')
    parser.add_argument('--no_training', action="store_false", dest="is_training", default=True,
                    help='Should we just evaluate?')
    parser.add_argument('--model_dir', type=lambda x: is_valid_file(parser, x), default='/tmp/bayes_by_backprop',
                    help='The model directory.')

    args = parser.parse_args()

    run(args)
