


'''
Caculating Accuracy
'''
class ScoreManager():
    def __init__(self) -> None:
        pass
    
        
    def hit_rate(self, y_true, y_pred, k):
        '''
        Recording hit if target is in top-k items
        
        y_true (Batch,): label of output - index type    
        y_pred (Batch,Items): prediction output
        k : number to choose top # items
        '''
        
        y_pred = y_pred.numpy()
        length = len(y_pred)
        
        hit = 0
        for i in range(length):
            indices = (-y_pred[i]).argsort()[:k]
            if y_true[i] in indices:
                hit += 1
                
        hit_rate = hit / length
        
        return hit_rate
    
    def mrr(self, y_true, y_pred, k):
        '''
        Calculating MRR
        '''
        
        