import tensorflow as tf
from tensorflow.keras import Model, Sequential
from Models.GRU_Block import GRU_Block

class GRU_Layer(Model):
    def __init__(self, n_dim, m_seq):
        super().__init__()
        self.n_dim = n_dim
        self.m_seq = m_seq
        
        self.gru_block = GRU_Block(n_dim)
        
    def call(self, x):
        ### BatchSize x n_dim x seq        
        output = None
        x_before = None
    
        for i in range(self.m_seq):
            if (i == 0):
                x_before = self.gru_block(x[:,:,i])
                output = tf.expand_dims(x_before, 2)
            else:
                x_before = self.gru_block(x[:,:,i], x_before)
                x_exp = tf.expand_dims(x_before, 2)
                output = tf.concat([output, x_exp], 2)
        
        return output