import numpy as np

def softmax(x, temperature):
    """
    Compute softmax values for each sets of scores in x.
    x: logits, a np array
    temperature: hyperparameter
    """
    x = x / temperature
    e_x = np.exp(x - np.max(x))
    
    return e_x / e_x.sum() 

