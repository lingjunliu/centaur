import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    A = torch.tensor(input_dict["A"])
    B = torch.tensor(np.eye(A.shape[0]), dtype=A.dtype) if "B" not in input_dict or input_dict["B"] is None else torch.tensor(input_dict["B"])
    k = input_dict.get("k", 1)
    n = k if "n" not in input_dict or input_dict["n"] is None else input_dict["n"]
    X = torch.rand(A.shape[0], n, dtype=A.dtype) if "X" not in input_dict else torch.tensor(input_dict["X"], dtype=A.dtype)
    iK = torch.tensor(input_dict["iK"]) if "iK" in input_dict else None
    tol = input_dict.get("tol", np.finfo(A.dtype).eps ** 0.5)
    largest = input_dict.get("largest", True)
    method = input_dict.get("method", "ortho")
    niter = input_dict.get("niter", None)
    tracker = input_dict.get("tracker", None)
    ortho_iparams = input_dict.get("ortho_iparams", None)
    ortho_fparams = input_dict.get("ortho_fparams", None)
    ortho_bparams = input_dict.get("ortho_bparams", None)
    
    if not cpu:
        A = A.cuda()
        if B is not None:
            B = B.cuda()
        if X is not None:
            X = X.cuda()
        if iK is not None:
            iK = iK.cuda()
    
    E, X = torch.lobpcg(A, k=k, B=B, X=X, n=n, iK=iK, niter=niter, tol=tol, largest=largest, method=method, tracker=tracker, ortho_iparams=ortho_iparams, ortho_fparams=ortho_fparams, ortho_bparams=ortho_bparams)
    
    if not cpu:
        E = E.cpu()
        X = X.cpu()
    
    return {"E": E.numpy(), "X": X.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import numpy as np

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        A = tf.constant(input_dict["A"])
        B = tf.constant(input_dict["B"]) if "B" in input_dict else None
        X = tf.constant(input_dict["X"]) if "X" in input_dict else None
        iK = tf.constant(input_dict["iK"]) if "iK" in input_dict else None
        k = input_dict.get("k", None)
        n = input_dict.get("n", None)
        tol = input_dict.get("tol", np.finfo(A.dtype).eps ** 0.5)
        largest = input_dict.get("largest", True)
        method = input_dict.get("method", "ortho")
        niter = input_dict.get("niter", None)
        ortho_iparams = input_dict.get("ortho_iparams", None)
        ortho_fparams = input_dict.get("ortho_fparams", None)
        ortho_bparams = input_dict.get("ortho_bparams", None)

        if B is None:
            if k is None:
                k = 1
            if X is None:
                w, v = tf.linalg.eig(A)
                if largest:
                    eigenvalues = tf.math.real(w)
                    eigenvectors = tf.math.real(v)
                    indices = tf.argsort(eigenvalues, direction='DESCENDING')
                    eigenvalues = tf.gather(eigenvalues, indices[:k])
                    eigenvectors = tf.gather(eigenvectors, indices[:k], axis=1)
                else:
                    eigenvalues = tf.math.real(w)
                    eigenvectors = tf.math.real(v)
                    indices = tf.argsort(eigenvalues, direction='ASCENDING')
                    eigenvalues = tf.gather(eigenvalues, indices[:k])
                    eigenvectors = tf.gather(eigenvectors, indices[:k], axis=1)
                E = eigenvalues.numpy()
                X = eigenvectors.numpy()
            else:
                w, v = tf.linalg.eig(A)
                if largest:
                    eigenvalues = tf.math.real(w)
                    eigenvectors = tf.math.real(v)
                    if k is None:
                        k = 1
                    indices = tf.argsort(eigenvalues, direction='DESCENDING')
                    eigenvalues = tf.gather(eigenvalues, indices[:k])
                    eigenvectors = tf.gather(eigenvectors, indices[:k], axis=1)
                else:
                    eigenvalues = tf.math.real(w)
                    eigenvectors = tf.math.real(v)
                    indices = tf.argsort(eigenvalues, direction='ASCENDING')
                    eigenvalues = tf.gather(eigenvalues, indices[:k])
                    eigenvectors = tf.gather(eigenvectors, indices[:k], axis=1)

                E = eigenvalues.numpy()
                X = eigenvectors.numpy()


        else:
            raise NotImplementedError("Tensorflow version with B matrix not implemented. Use torch version instead.")

    return {"E": E, "X": X}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[4.0, 1.0], [1.0, 4.0]], dtype=np.float32),
        "B": None,
        "k": 1,
        "X": None,
        "n": None,
        "tol": None,
        "largest": True,
        "method": "ortho",
        "niter": None
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["E"], tf_result["E"], atol=A_TOL), "Eigenvalues do not match"
    assert np.allclose(torch_result["X"], tf_result["X"], atol=A_TOL), "Eigenvectors do not match"

    print("Success")

if __name__ == "__main__":
    main()