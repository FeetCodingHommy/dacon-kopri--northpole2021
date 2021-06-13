import tensorflow as tf


class WeightMultiply(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(WeightMultiply, self).__init__()
        self.w = tf.Variable(
            tf.ones([1], dtype="float")/2,
            trainable=True,
            dtype="float"
        )
        self.one = tf.Variable(
            tf.ones([1], dtype="float"),
            trainable=False,
            dtype="float"
        )
    
    def call(self, input_layer1, input_layer2):
        weight_rev = tf.math.subtract(self.one, self.w)
        return tf.math.multiply(self.w, input_layer1), tf.math.multiply(weight_rev, input_layer2)
