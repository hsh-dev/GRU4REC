from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import sigmoid, tanh

class GRU_Layer(Model):
    def __init__(self, n_dim):
        super().__init__()
        self.n_dim = n_dim
        
        self.r_W = Dense(n_dim)
        self.r_U = Dense(n_dim)
        self.r_sigmoid = sigmoid()
        
        
    def call(self, x, h):
        r = self.r_sigmoid(self.r_W(x) + self.r_U(h))
        
        return r
        
    