# TODO: Fix

import torch
import tensorflow as tf


def torch_version(input, cpu=True):
    # input
    x1 = input["embed_dim"]
    x2 = input["num_heads"]
    x3 = input["dropout"]
    
    # class instance
    func_obj = torch.nn.MultiheadAttention(embed_dim=x1, num_heads=x2, dropout=x3, batch_first=True)

    # second set of inputs
    x4 = torch.tensor(input["query"])
    x5 = torch.tensor(input["key"])
    x6 = torch.tensor(input["value"])
    
    
    if not cpu:
        x4 = x4.cuda()
        x5 = x5.cuda()
        x6 = x6.cuda()
        func_obj = func_obj.cuda()

    # output
    y = func_obj(query=x4, key=x5, value=x6)

    if not cpu:
        y = (y[0].cpu(), y[1].cpu())

    return {"attn_output": y[0].detach().numpy()}#, "attn_output_weights": y[1].detach().numpy()}


def tensorflow_version(input, cpu=True):
    if cpu:
        device_string = "/cpu"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # input
        x1 = input["embed_dim"]
        x2 = input["num_heads"]
        x3 = input["dropout"]
        
        x4 = tf.constant(input["query"])
        x5 = tf.constant(input["key"])
        x6 = tf.constant(input["value"])

        # output
        y = tf.keras.layers.MultiHeadAttention(num_heads=x2, key_dim=x1, value_dim=x1, dropout=x3)(query=x4, value=x6, key=x5)

        return {"attn_output": y[0].numpy()}#, "attn_output_weights": y[1].numpy()}
