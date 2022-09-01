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
        
        # neg_square = K.square(negative_list)
        # pos_square = K.square(positive_list)    
        # neg_square_sum = K.sum(neg_square)
        # pos_square_sum = K.sum(pos_square)
        # sample_size = neg_square.shape[0] * neg_square.shape[1]
        # reg_loss = (neg_square_sum - pos_square_sum)
        # loss = K.mean(cal) + reg_loss
        
        loss = K.mean(cal)
        
        return loss