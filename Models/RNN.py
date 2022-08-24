from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from Models.GRU_Block import GRU_Block

class RNN(Model):
    def __init__(self, n_dim, seq_dim, gru_num, out_dim):
        super().__init__()
        '''
        n_dim : hidden layer
        seq_dim : sequence count of gru
        out_dim : output dimension of feedforward layer
        '''
        self.n_dim = n_dim
        self.seq_dim = seq_dim
        self.gru_num = gru_num
        self.out_dim = out_dim
        
        self.gru_layer_1 = GRU_Layer(n_dim, seq_dim)
        self.gru_layer_2 = GRU_Layer(n_dim, seq_dim)
        self.gru_layer_3 = GRU_Layer(n_dim, seq_dim)
        self.gru_layer_4 = GRU_Layer(n_dim, seq_dim)
        
        self.fc_layer = Dense(out_dim)
        
    def call(self, x):
        shortcut = x
        
        x = self.gru_layer_1(x)
        x = self.gru_layer_2(x + shortcut)
        x = self.gru_layer_3(x)
        x = self.gru_layer_4(x + shortcut)
        
        ### fc_layer
        
        return x