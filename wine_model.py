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

def absolute_difference(y1, y2):
    res = tf.reduce_mean(tf.cast(tf.abs(tf.argmax(y1, output_type=tf.int32, axis=1) -tf.argmax(y2, output_type=tf.int32, axis=1)), tf.float32))
    return res



def get_compiled_model():
    alpha = .0
    reg=.008
    # activation = tf.keras.layers.LeakyReLU(alpha=alpha)
    activation='relu'
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(11,)),
        tf.keras.layers.Dense(192, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(reg)),
        tf.keras.layers.Dropout(rate=.5),
        tf.keras.layers.Dense(96, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(reg)),
        tf.keras.layers.Dense(48, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(reg)),
        tf.keras.layers.Dense(11, activation='softmax')
    ])

    # Ensure TF doesn't allocate all VRAM at start-up, so I can run multiple program instances concurrently
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))

    optimizer = tf.train.AdamOptimizer()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[absolute_difference])
    return model


def normalize_dataset(X_train, X_test):
    features_avg = X_train.mean(axis=0)
    features_std = X_train.std(axis=0)
    return (X_train-features_avg)/features_std, (X_test-features_avg)/features_std

def min_max_normalize_dataset(X_train, X_test):
    features_min = X_train.min(axis=0)
    features_max = X_train.max(axis=0)
    return (X_train-features_min)/(features_max-features_min), (X_test-features_min)/(features_max-features_min)


def main():
    # Just comment the next line out to disable eager execution
    # tf.enable_eager_execution()

    """
    Set use_fit to True to optimize by calling tf.keras.models.Sequential.fit(),
    set to False to use tfe.GradientTape() instead. Note that in order to use tfe.Gradient.tape(),
    eager execution must be enabled
    """
    epochs = 1000
    batch_size = 128
    dataset_folder = '/home/fanta/.kaggle/competitions/uci-wine-quality-dataset'

    # Load dataset and convert it to numpy arrays
    data = load_data(dataset_folder)
    X = data.iloc[:, 0:11].values.astype(np.float32)
    y = data.iloc[:, 11].values.astype(np.int32)

    y = tf.keras.utils.to_categorical(y, num_classes=11)

    n_splits = 5
    dataset_splitter = model_selection.KFold(n_splits=n_splits, shuffle=True)

    start = time.time()

    # Initialise metrics to be plotted
    avg_loss_by_epoch = np.zeros((epochs), dtype=np.float)
    avg_error_by_epoch = np.zeros((epochs), dtype=np.float)
    val_avg_loss_by_epoch = np.zeros((epochs), dtype=np.float)
    val_avg_error_by_epoch = np.zeros((epochs), dtype=np.float)
    # Go around every fold
    for fold, (train_idx, val_idx) in enumerate(dataset_splitter.split(X, y)):
        print('\nTraining fold {} of {}.'.format(fold+1, n_splits))
        model = get_compiled_model()
        # Split dataset between training and validation set
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        X_train, X_val = min_max_normalize_dataset(X_train, X_val)
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

# 200 epochs
# Final val. loss=1.2299661381506835   final val. error=0.5689313082391819   (averages across folds)
# It took 87.59770274162292 seconds

# 200 epochs, deeper network
# Final val. loss=1.2304934671804009   final val. error=0.5622761750204474   (averages across folds)
# It took 92.3689968585968 seconds

# dropout .3 at 4th layer
# Final val. loss=1.1370621089066315   final val. error=0.5615007689203808   (averages across folds)
# It took 100.11282062530518 seconds

# droput .5
# Final val. loss=1.1590469943171178   final val. error=0.5683999272185376   (averages across folds)
# It took 100.55076003074646 seconds

# RELU and normalization [0, 1]
# Final val. loss=1.050925898971086   final val. error=0.488251948486815   (averages across folds)
# It took 102.0314393043518 seconds

# L2 regularizers for weights on every dense layer
# Final val. loss=1.1740582234589088   final val. error=0.5421094300626239   (averages across folds)
# It took 110.20939540863037 seconds

# L2 reg. with l=.008
# Final val. loss=1.1668800430232977   final val. error=0.5352356844490612   (averages across folds)
# It took 108.70994210243225 seconds

# 400 epochs
# Final val. loss=1.152037846473799   final val. error=0.5199142493920655   (averages across folds)
# It took 217.8574903011322 seconds

# batch size = 128
# Final val. loss=1.155156646635937   final val. error=0.5122517398296567   (averages across folds)
# It took 115.09231066703796 seconds

# 1000 epochs
# Final val. loss=1.136638939387792   final val. error=0.5002537990202389   (averages across folds)
# It took 280.46708631515503 seconds

# SGD lr=0.05
# Final val. loss=1.1638371995259298   final val. error=0.5316476401120169   (averages across folds)
# It took 463.5820310115814 seconds

# Adagrad lr=.1
# Final val. loss=1.1477592689852358   final val. error=0.5132653062366341   (averages across folds)
# It took 471.4418022632599 seconds

# Adadelta
# rubbish

# Adam 1000 epochs, default lr
# Final val. loss=1.1360728829271847   final val. error=0.49822145353021163   (averages across folds)
# It took 531.8769500255585 seconds

# TODO try with TF layers instead of Keras. Submit a prediction to Kaggle.


