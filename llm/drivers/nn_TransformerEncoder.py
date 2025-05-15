import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    encoder_layer = input_dict["encoder_layer"]
    num_layers = input_dict["num_layers"]
    norm = input_dict.get("norm", None)
    enable_nested_tensor = input_dict.get("enable_nested_tensor", True)
    mask_check = input_dict.get("mask_check", True)
    
    encoder_layer_instance = nn.TransformerEncoderLayer(d_model=encoder_layer['d_model'], nhead=encoder_layer['nhead'])
    transformer_encoder = nn.TransformerEncoder(encoder_layer_instance, num_layers, norm, enable_nested_tensor, mask_check)

    src = torch.tensor(input_dict["src"])
    mask = input_dict.get("mask", None)
    if mask is not None:
        mask = torch.tensor(mask)
    src_key_padding_mask = input_dict.get("src_key_padding_mask", None)
    if src_key_padding_mask is not None:
        src_key_padding_mask = torch.tensor(src_key_padding_mask)
    is_causal = input_dict.get("is_causal", None)
    
    if not cpu:
        transformer_encoder = transformer_encoder.cuda()
        src = src.cuda()
        if mask is not None:
            mask = mask.cuda()
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.cuda()
    
    out = transformer_encoder(src, mask, src_key_padding_mask, is_causal)
    
    if not cpu:
        out = out.cpu()
    
    return {"result": out.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    
    encoder_layer = input_dict["encoder_layer"]
    num_layers = input_dict["num_layers"]
    
    src = tf.constant(input_dict["src"])
    mask = input_dict.get("mask", None)
    src_key_padding_mask = input_dict.get("src_key_padding_mask", None)
    is_causal = input_dict.get("is_causal", None)

    d_model = encoder_layer['d_model']
    nhead = encoder_layer['nhead']
    
    def scaled_dot_product_attention(q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.float32)
            scaled_attention_logits += (mask * -1e9)
            
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights
    
    def mlp(x, d_model):
        x = tf.keras.layers.Dense(4 * d_model, activation='relu')(x)
        x = tf.keras.layers.Dense(d_model)(x)
        return x
    
    def encoder_layer_tf(x, mask):
        q = tf.keras.layers.Dense(d_model)(x)
        k = tf.keras.layers.Dense(d_model)(x)
        v = tf.keras.layers.Dense(d_model)(x)
        
        q_split = tf.split(q, num_or_size_splits=nhead, axis=-1)
        k_split = tf.split(k, num_or_size_splits=nhead, axis=-1)
        v_split = tf.split(v, num_or_size_splits=nhead, axis=-1)
        
        attention_outputs = []
        for i in range(nhead):
            attention_output, _ = scaled_dot_product_attention(q_split[i], k_split[i], v_split[i], mask)
            attention_outputs.append(attention_output)
        
        attention_output = tf.concat(attention_outputs, axis=-1)
        attention_output = tf.keras.layers.Dense(d_model)(attention_output)
        
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        mlp_output = mlp(x, d_model)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + mlp_output)
        return x
    
    output = src
    
    extended_attention_mask = None
    if mask is not None:
        seq_len = tf.shape(src)[0]
        extended_attention_mask = tf.reshape(mask, (1, 1, seq_len, seq_len))
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    
    for _ in range(num_layers):
        output = encoder_layer_tf(output, extended_attention_mask)
    
    return {"result": output.numpy()}

def main():
    import torch
    A_TOL = 0.01
    
    encoder_layer_config = {'d_model': 512, 'nhead': 8}

    input_data = {
        "encoder_layer": encoder_layer_config,
        "num_layers": 2,
        "src": np.random.rand(10, 32, 512).astype(np.float32),
        "mask": np.random.randint(0, 2, size=(10, 10)).astype(np.float32),
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()