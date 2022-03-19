import os
import pickle
import argparse
import tensorflow as tf
import tensorflow.keras as keras
from model import BaseModel, SimpleTestModel


def build_dataset(data_path):
    with open(data_path, 'rb') as f:
        datafile = pickle.load(f)
        states = [item[0] for item in datafile]
        operations = [item[1] for item in datafile]
        labels = [item[2] for item in datafile]
    return tf.data.Dataset.from_tensor_slices((states, operations, labels))


# @tf.function
def train_step(states, operations, labels, model, loss_fn, loss_accumulator, optimizer, acc_metric):
    labels = tf.math.abs(labels) ** 2
    with tf.GradientTape() as tape:
        outputs = model(states, operations, training=True)
        # print(states.numpy())
        # print(outputs.numpy())
        # assert False
        loss = loss_fn(labels, outputs)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    loss_accumulator.update_state(loss)
    acc_metric.update_state(labels, outputs)


@tf.function
def test_step(states, operations, labels, model, acc_metric):
    labels = tf.math.abs(labels) ** 2
    outputs = model(states, operations, training=False)
    acc_metric.update_state(labels, outputs)


def main(args):
    train_dataset = build_dataset(args.train_path).batch(args.batch_size)
    test_dataset = build_dataset(args.test_path).batch(args.batch_size)
    
    model = BaseModel(out_dim=32)
    # model = SimpleTestModel()
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = keras.losses.CategoricalCrossentropy()
    train_loss = keras.metrics.Mean(name='train_loss')
    train_acc = keras.metrics.CosineSimilarity(name='train_acc')
    test_acc = keras.metrics.CosineSimilarity(name='test_acc')

    for epo in range(args.epochs):
        train_loss.reset_states()
        train_acc.reset_states()
        test_acc.reset_states()

        for states, operations, labels in train_dataset:
            train_step(states, operations, labels, model, loss_fn, loss_accumulator=train_loss, optimizer=optimizer, acc_metric=train_acc)

        for states, operations, labels in test_dataset:
            test_step(states, operations, labels, model, acc_metric=test_acc)
        print(
            f'Epoch {epo}, '
            f'Loss: {train_loss.result()}, '
            f'Train Accuracy (Cosine Similarity): {train_acc.result():6f}, '
            f'Test Accuracy (Cosine Similarity): {test_acc.result():6f}'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', default='data/trainset.pkl', type=str)
    parser.add_argument('--test-path', default='data/testset.pkl', type=str)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=str)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)