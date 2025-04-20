import torch
import tensorflow as tf

def torch_version(input, cpu=True):
    # input
    x = torch.tensor(input["input"])
    
    if not cpu:
        x = x.cuda()
        
    # output
    y = torch.ravel(x)
    
    if not cpu:
        y = y.cpu()          
    
    return {"ravel": y.numpy()}


def tensorflow_version(input, cpu=True):
    if cpu:
        device_string = "/cpu"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # input
        x = tf.constant(input["input"])

        # output
        y = tf.experimental.numpy.ravel(x)
        
        return {"ravel": y.numpy()}