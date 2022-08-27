from DataModule.DataLoader import DataLoader
from TrainModule.TrainManager import TrainManager

import tensorflow as tf

from Models.GRU_Block import GRU_Block
from Models.GRU_Layer import GRU_Layer

from DataModule.DataLoader import DataLoader


if __name__ == "__main__":
    
    # gru = GRU_Layer(8, 3)
    # input = tf.random.normal([1, 8, 3], 0, 1, tf.float32)
    # print("model test")
    # output = gru(input)
    # print(output)
    
    dataloader = DataLoader()
    dataloader.make_session()
    # movie = dataloader.get_movie(10)
    