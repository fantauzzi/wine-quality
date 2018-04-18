import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import model_selection
import tensorflow as tf
import tensorflow.contrib.eager as tfe

"""
Check the beginning of main() for parameters
"""


def load_data(folder):
    file_name = os.path.join(folder, 'winequality-data.csv')
    if os.path.isfile(file_name):
        data = pd.read_csv(file_name)
    else:
        print('File {} not found.'.format(file_name, folder))
        print('Dataset can be downloaded from https://www.kaggle.com/c/uci-wine-quality-dataset/data')
        exit(1)
    # solutions = pd.read_csv(os.path.join(folder, 'winequality-solution-input.csv'))
    return data


def train_input_fn(features, labels, batch_size):
    features_tensor = tf.constant(features)
    labels_tensor = tf.constant(labels)
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((features_tensor, labels_tensor))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(len(labels)).repeat(count=1).batch(batch_size)

    # Return the dataset.
    return dataset


def loss(model, X, y):
    logits = model(X)
    the_loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
    return the_loss


def grad(model, inputs, targets):
    with tfe.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


def train(model, X, y, batch_size, epochs):
    train_ds = train_input_fn(X, y, batch_size=batch_size)
    optimizer = tf.train.AdamOptimizer()

    loss_by_epoch = []
    accuracy_by_epoch = []

    for epoch in range(epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_error = tfe.metrics.Mean()
        for batch, (batch_X, batch_y) in enumerate(tfe.Iterator(train_ds)):
            grads = grad(model, batch_X, batch_y)
            optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())
            epoch_loss_avg(loss(model, batch_X, batch_y))
            correct_prediction = tf.equal(tf.argmax(model(batch_X), axis=1, output_type=tf.int32), batch_y)
            epoch_error(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
        print('Epoch {}:  loss={}  accuracy={}'.format(epoch, epoch_loss_avg.result(), epoch_error.result()))
        loss_by_epoch.append(epoch_loss_avg.result())
        accuracy_by_epoch.append(epoch_error.result())

    return loss_by_epoch, accuracy_by_epoch


def get_compiled_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(11,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(11, activation='softmax')
    ])

    optimizer = tf.train.AdamOptimizer()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[tf.losses.absolute_difference])
    return model

def main():
    # Just comment the next line out to disable eager execution
    # tf.enable_eager_execution()

    """
    Set use_fit to True to optimize by calling tf.keras.models.Sequential.fit(),
    set to False to use tfe.GradientTape() instead. Note that in order to use tfe.Gradient.tape(),
    eager execution must be enabled
    """
    epochs = 400
    batch_size = 64
    dataset_folder = '/home/fanta/.kaggle/competitions/uci-wine-quality-dataset'

    # Load dataset and convert it to numpy arrays
    data = load_data(dataset_folder)
    X = data.iloc[:, 0:11].values.astype(np.float32)
    y = data.iloc[:, 11].values.astype(np.int32)

    y = tf.keras.utils.to_categorical(y, num_classes=11)


    n_splits = 5
    dataset_splitter = model_selection.KFold(n_splits=n_splits, shuffle=True)

    start = time.time()

    # Loop around the estimators
    avg_loss_by_epoch = np.zeros((epochs), dtype=np.float)
    avg_error_by_epoch = np.zeros((epochs), dtype=np.float)
    val_avg_loss_by_epoch = np.zeros((epochs), dtype=np.float)
    val_avg_error_by_epoch = np.zeros((epochs), dtype=np.float)
    for fold, (train_idx, val_idx) in enumerate(dataset_splitter.split(X, y)):
        print('\nTraining fold {} of {}.'.format(fold, n_splits))
        model = get_compiled_model()
        # Split dataset between training and validation set
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        history = model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), shuffle=False, epochs=epochs, batch_size=batch_size, verbose=2)
        # Compute loss and accuracy for this fold
        loss_by_epoch = history.history['loss']
        error_by_epoch = history.history['absolute_difference']
        val_loss_by_epoch = history.history['val_loss']
        val_error_by_epoch = history.history['val_absolute_difference']
        # Update overall loss and accuracy (average across all folds)
        avg_error_by_epoch += error_by_epoch
        avg_loss_by_epoch += loss_by_epoch
        val_avg_error_by_epoch += val_error_by_epoch
        val_avg_loss_by_epoch += val_loss_by_epoch
    avg_error_by_epoch /= n_splits
    avg_loss_by_epoch /= n_splits
    val_avg_error_by_epoch /= n_splits
    val_avg_loss_by_epoch /= n_splits
    print('\nFinal val. loss={}   final val. error={}   (averages across folds)'.format(val_avg_loss_by_epoch[-1], val_avg_error_by_epoch[-1]))



    end = time.time()
    print('It took {} seconds'.format(end - start))

    # Chart loss and error
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(avg_loss_by_epoch, label='Training')
    axes[0].plot(val_avg_loss_by_epoch, label='Validation')
    axes[0].legend()

    axes[1].set_ylabel("Absolute Error", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(avg_error_by_epoch, label='Training')
    axes[1].plot(val_avg_error_by_epoch, label='Validation')
    axes[1].legend()

    plt.show()


if __name__ == '__main__':
    main()

# 200 epochs
# Final val. loss1.240027957177968   final val. error=0.10588621345913296
# It took 91.67309617996216 seconds

# 400 epochs
# Final val. loss=1.1593427044662927   final val. error=0.10552638304637627   (averages across folds)
# It took 180.43536710739136 seconds