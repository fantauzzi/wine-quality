import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Model


# Using Keras functional API
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.inp = tf.keras.layers.Input(shape=(8,))
        self.fc1 = tf.keras.layers.Dense(32)
        self.fc2 = tf.keras.layers.Dense(10)

    def __call__(self, inputs, trainig=False):
        y = self.inp(inputs)
        y = self.fc1(y)
        y = self.fc2(y)
        return y


if __name__ == '__main__':
    # tf.enable_eager_execution()

    # Just a random dataset, to try out the code
    X = np.random.rand(512, 8)
    y = np.random.randint(0, 9, size=(512))

    model = MyModel()

    """
    inputs=tf.keras.layers.Input(shape=(8,))
    outputs = tf.keras.layers.Dense(32)(inputs)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(outputs)

    model = tf.keras.Model(inputs=inputs, outputs = outputs)
    """

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.train.AdadeltaOptimizer(),
                  metric=['accuracy'])

    model.fit(X, y, batch_size=64, epochs=1, verbose=2)
