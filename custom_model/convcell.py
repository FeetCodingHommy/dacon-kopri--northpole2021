import tensorflow as tf


class ConvLSTMCell(tf.keras.Model):
    def __init__(self, hidden_dim, kernel_size, bias, activation="tanh"):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.kernel_size = kernel_size
        self.bias = bias
        
        self.conv = tf.keras.layers.Conv2D(
            filters = 4 * self.hidden_dim,
            kernel_size = self.kernel_size,
            padding = 'same',
            use_bias = self.bias,
            activation = activation
        )
        
    def call(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = tf.concat([input_tensor, h_cur], axis=3)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = tf.split(combined_conv, num_or_size_splits=4, axis=-1)
        i = tf.keras.activations.sigmoid(cc_i)
        f = tf.keras.activations.sigmoid(cc_f)
        o = tf.keras.activations.sigmoid(cc_o)
        g = tf.keras.activations.tanh(cc_g)
        
        c_next = f*c_cur+i*g
        h_next = o*tf.keras.activations.tanh(c_next)
        
        return h_next, c_next
        
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (tf.zeros([batch_size, height, width, self.hidden_dim]),
                tf.zeros([batch_size, height, width, self.hidden_dim]))


class ConvGRUCell(tf.keras.Model):
    def __init__(self, hidden_dim, kernel_size, bias, activation="tanh"):
        super(ConvGRUCell, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.kernel_size = kernel_size
        self.bias = bias
        
        self.conv_xh = tf.keras.layers.Conv2D(
            filters = 2 * self.hidden_dim,
            kernel_size = self.kernel_size,
            padding = 'same',
            use_bias = self.bias,
            activation = activation
        )

        self.conv_xr = tf.keras.layers.Conv2D(
            filters = 1 * self.hidden_dim,
            kernel_size = self.kernel_size,
            padding = 'same',
            use_bias = self.bias,
            activation = activation
        )
        
    def call(self, input_tensor, cur_state):
        h_cur = cur_state[0]
        combined = tf.concat([input_tensor, h_cur], axis=3)
        combined_conv_rz = self.conv_xh(combined)
        cc_r, cc_z = tf.split(combined_conv_rz, num_or_size_splits=2, axis=-1)
        
        r = tf.keras.activations.sigmoid(cc_r)
        z = tf.keras.activations.sigmoid(cc_z)
        
        combined2 = tf.concat([input_tensor, r*h_cur], axis=3)
        x_reset = self.conv_xr(combined2)
        h_tilde = tf.keras.activations.tanh(x_reset)
        
        h_next = (1-z)*h_cur + z*h_tilde
        
        return h_next
        
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return tf.zeros([batch_size, height, width, self.hidden_dim])


class ConvRNNCell(tf.keras.Model):
    def __init__(self, hidden_dim, kernel_size, bias, activation="tanh"):
        super(ConvRNNCell, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.kernel_size = kernel_size
        self.bias = bias
        
        self.conv = tf.keras.layers.Conv2D(
            filters = self.hidden_dim,
            kernel_size = self.kernel_size,
            padding = 'same',
            use_bias = self.bias,
            activation = activation
        )
        
    def call(self, input_tensor, cur_state):
        h_cur = cur_state[0]
        combined_conv = self.conv(tf.concat([input_tensor, h_cur], axis=3))
        h_next = tf.keras.activations.tanh(combined_conv)
        
        return h_next
        
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return tf.zeros([batch_size, height, width, self.hidden_dim])
