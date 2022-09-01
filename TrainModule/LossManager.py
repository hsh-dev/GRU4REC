import tensorflow as tf
import keras.backend as K


class LossManager():    
    '''
    Loss Functions
    '''
    def __init__(self) -> None:
        pass
    
    @tf.function
    def cross_entropy_loss(self, y_true, y_pred):
        cce = tf.keras.losses.CategoricalCrossentropy()
        err = cce(y_true, y_pred)
        
        return err        
    
    @tf.function
    def top_1_ranking_loss(self, y_true_idx, y_pred):        
        negative_list = tf.gather(y_pred, indices = y_true_idx, axis = 1)

        y_true_idx = tf.expand_dims(y_true_idx, axis = 1)
        positive_list = tf.gather(y_pred, indices = y_true_idx, axis = 1, batch_dims=1)
        
        cal = K.sigmoid(negative_list - positive_list)
        # reg = K.square(negative_list)
        # reg_pos = K.square(positive_list)    
    
        loss = K.mean(cal)
        
        return loss