import enum
from DataModule.DataLoader import DataLoader
from TrainModule.TrainManager import TrainManager

import tensorflow as tf

from Models.GRU_Block import GRU_Block
from Models.GRU_Layer import GRU_Layer

from Models.RNN import RNN

from DataModule.DataLoader import DataLoader

HIDDEN_DIM = 10
BATCH_SIZE = 16

if __name__ == "__main__":
    
    rnn = RNN(HIDDEN_DIM, BATCH_SIZE)
    
    # gru = GRU_Layer(8, 3)
    # input = tf.random.normal([10, 8], 0, 10, tf.int32)
    # print("model test")
    # output = rnn(input)
    
    dataloader = DataLoader()
    dataset = dataloader.get_train_set()
    
    for idx, sample in enumerate(dataset):
        x, y = sample
        
        output = rnn(x)

    