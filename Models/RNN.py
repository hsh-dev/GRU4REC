from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Softmax
from Models.GRU_Layer import GRU_Layer

class RNN(Model):
    def __init__(self, n_dim, seq_dim):
        super().__init__()
        '''
        n_dim : hidden layer dimension = embedding dimension
        seq_dim : sequence count - minibatch length
        out_dim : output dimension of feedforward layer
        '''
        self.n_dim = n_dim
        self.seq_dim = seq_dim
        # self.gru_num = gru_num
        self.out_dim = 128
        
        self.embedding_layer = Embedding(1, n_dim)
        
        self.gru_layer_1 = GRU_Layer(n_dim, seq_dim)
        self.gru_layer_2 = GRU_Layer(n_dim, seq_dim)
        self.gru_layer_3 = GRU_Layer(n_dim, seq_dim)
        self.gru_layer_4 = GRU_Layer(n_dim, seq_dim)
        
        self.fc_layer = Dense(self.out_dim)
        
        self.softmax = Softmax()
        
    def call(self, x):
        x = self.embedding_layer(x)
        
        shortcut = x
        
        x = self.gru_layer_1(x)
        x = self.gru_layer_2(x + shortcut)
        x = self.gru_layer_3(x)
        x = self.gru_layer_4(x + shortcut)
        
        x = self.fc_layer(x)
        
        x = self.softmax(x)
        
        return x