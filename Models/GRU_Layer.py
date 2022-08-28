import tensorflow as tf
from tensorflow.keras import Model, Sequential
from Models.GRU_Block import GRU_Block

class GRU_Layer(Model):
    def __init__(self, n_dim, m_seq):
        super().__init__()
        self.n_dim = n_dim  ## hidden layer dimension
        self.m_seq = m_seq  ## count of gru block
        
        self.gru_block = GRU_Block(n_dim)
        
    def call(self, x):
        ### BatchSize x n_dim      
        x_out = None

        for i in range(self.m_seq):
            if (i == 0):
                x_out = self.gru_block(x)
            else:
                x_out = self.gru_block(x, x_out)

        return x_out