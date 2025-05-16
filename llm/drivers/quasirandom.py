import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    seq_len = input_dict["seq_len"]
    seed = input_dict.get("seed", None)
    
    if seed is not None:
        seed = int(torch.tensor(seed).item())

    if not cpu:
        pass
            
    engine = torch.quasirandom.SobolEngine(dimension=input_dict["dimension"], scramble=input_dict.get("scramble", False), seed=seed)
    result = engine.draw(n=seq_len)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    seq_len = input_dict["seq_len"]
    dimension = input_dict["dimension"]
    seed_value = input_dict.get("seed", None)
    scramble = input_dict.get("scramble", False)

    if seed_value is None:
        seed = np.random.randint(0, 2**32 - 1)
    else:
        seed = seed_value

    def generate_sobol(num_points, dimension, seed_value):
        sobol_seq = np.zeros((num_points, dimension), dtype=np.float32)
        v = np.zeros((dimension, 32), dtype=np.int32)

        # Initialize direction vectors (V)
        for i in range(1, 32):
            for j in range(dimension):
                v[j, i-1] = int(bin(j + 1)[2:].zfill(32)[i-1])

        # Generate Sobol sequence
        x = np.zeros((num_points, dimension), dtype=np.int32)
        for k in range(num_points):
            i = 0
            value = k
            while value & 1:
                value >>= 1
                i += 1

            if k > 0:
                x[k] = np.bitwise_xor(x[k-1], v[:, i])


        sobol_seq = x / (2**32)

        if scramble:
            np.random.seed(seed_value)
            rand_matrix = np.random.uniform(low=0.0, high=1.0, size=dimension)
            sobol_seq = np.mod(sobol_seq + rand_matrix, 1.0)

        return sobol_seq

    result = generate_sobol(seq_len, dimension, seed)
    result = result.astype(np.float32)
    
    return {"result": result}

def main():
    A_TOL = 0.01
    # Example input
    input_data = {
        "dimension": 2,
        "seq_len": 10,
        "scramble": True,
        "seed": 123
    }

    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Assert to see if they are equal
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()