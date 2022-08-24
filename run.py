from DataModule.DataLoader import DataLoader
from TrainModule.TrainManager import TrainManager

import tensorflow as tf

from Model.GRU_Layer import GRU_Layer



if __name__ == "__main__":
    
    gru = GRU_Layer(8)
    
    input = tf.zeros([1, 8], tf.float32)

    print("model test")
    
    output = gru(input)
    
    print(output)
    
    