import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.math import multiply, subtract

class GRU_Block(Model):
    def __init__(self, n_dim):
        super().__init__()
        self.n_dim = n_dim
        
        self.r_W = Dense(n_dim)
        self.r_U = Dense(n_dim)

        self.h_tild_W = Dense(n_dim)
        self.h_tild_U = Dense(n_dim)

        self.z_W = Dense(n_dim)
        self.z_U = Dense(n_dim)
        
    def call(self, x, h = None):
        if h == None:
            h = tf.zeros_like(x)
            
        r = sigmoid(self.r_W(x) + self.r_U(h))
        z = sigmoid(self.z_W(x) + self.z_U(h))
        h_tild = tanh(self.h_tild_W(x) + self.h_tild_U(r*h))
        
        ones_vector = tf.ones_like(z)
        h_new = subtract(ones_vector, z) * h + z * h_tild
        
        return h_new
        
    