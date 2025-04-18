import numpy as np
import logging
import pytest
import torch

from generator.rules import dist_1_rev, dist_7_rev, dist_8_rev, dist_5_rev

## Abid, please add assertions with corresponding expectations here --Marcelo

@pytest.mark.unit
def test_dist_1():
    ## I don't understand why this is true.
    # res=rule_1({'a':None}, {'b':None})    
    # print(res)

    ## another example
    t1=np.random.rand(3,2,5)
    t2=np.random.rand(4,2)
    # res=rule_1({'a':t1}, {'b':t2})
    # print(res) ## this is 
    print(dist_1_rev({'a':t1}, {'b':t2}))
    # simulate -1@1 on t1
    t1=np.random.rand(2,2,5) # from 3 to 2
    print(dist_1_rev({'a':t1}, {'b':t2}))
    ## got worse. let us try the other direction
    t1=np.random.rand(4,2,5) # from 3 to 4
    print(dist_1_rev({'a':t1}, {'b':t2})) ## this is the new best <~~~
    ## keep going the same direction
    t1=np.random.rand(5,2,5) # from 4 to 5
    print(dist_1_rev({'a':t1}, {'b':t2})) ## this is worse. get back to the previous best
    t1=np.random.rand(4,2,5) # recovering the best candidate
    ## try mutating the other input. remove one axis
    t2=np.random.rand(4)
    print(dist_1_rev({'a':t1}, {'b':t2})) ## this is much worse (bigger distance). go the other way
    t2=np.random.rand(4,2,1)
    print(dist_1_rev({'a':t1}, {'b':t2})) ## awesome. this is the new best <~~~
    t2=np.random.rand(3,2,1)
    print(dist_1_rev({'a':t1}, {'b':t2})) ## worse
    t2=np.random.rand(5,2,1)
    print(dist_1_rev({'a':t1}, {'b':t2})) ## worse
    t2=np.random.rand(4,2,1) ## recovering
    t2=np.random.rand(4,1,1)
    print(dist_1_rev({'a':t1}, {'b':t2})) ## worse
    t2=np.random.rand(4,3,1)
    print(dist_1_rev({'a':t1}, {'b':t2})) ## worse
    t2=np.random.rand(4,2,1) ## recovering
    t2=np.random.rand(4,2,0) ## worse
    print(dist_1_rev({'a':t1}, {'b':t2})) 
    t2=np.random.rand(4,2,2) ## better. this is the new best <~~~
    print(dist_1_rev({'a':t1}, {'b':t2}))
    t2=np.random.rand(4,2,5) ## perfect. 
    print(dist_1_rev({'a':t1}, {'b':t2}))

@pytest.mark.unit
def test_dist_5():
    input = np.random.rand(3, 5)
    index = np.array([[0, 1, 2, 0]])
    assert dist_5_rev({'input': input}, {'index': index}) == 0, f"Rule 5 was not satisfied for input with shape {input.shape} and index with values within [{np.min(index)}, {np.max(index)}]"
    
    input = torch.full((2, 4), 2., dtype=int).numpy()
    index = torch.tensor([[2], [3]]).numpy()
    assert dist_5_rev({'input': input}, {'index': index}) != 0, f"Rule 5 was satisfied for input with shape {input.shape} and index with values within [{np.min(index)}, {np.max(index)}], but it should not have"

@pytest.mark.unit
def test_dist_7():
    t1=np.random.rand(2,2)
    t2=np.random.rand(1,2)
    assert dist_7_rev({'a':t1}, {'b':t2}) == 0, f"Rule 7 was not satisfied for args with shape {t1.shape} and {t2.shape}" # should be 0
    t1=np.random.rand(3,2)
    t2=np.random.rand(2,2)
    assert dist_7_rev({'a':t1}, {'b':t2}) > 0, f"Rule 7 was satisfied for args with shape {t1.shape} and {t2.shape}, but it should not have" # should be higher than 0
    t1=np.random.rand(2,100)
    t2=np.random.rand(50,2)
    assert dist_7_rev({'a':t1}, {'b':t2}) == 0, f"Rule 7 was not satisfied for args with shape {t1.shape} and {t2.shape}" # should be 0
    t1=np.random.rand(3,100)
    t2=np.random.rand(100,2)
    assert dist_7_rev({'a':t1}, {'b':t2}) > 0, f"Rule 7 was satisfied for args with shape {t1.shape} and {t2.shape}, but it should not have" # should be higher than 0

@pytest.mark.unit
def test_dist_8():
    # Positive
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.int32)} # int32
    arg2 = {"other_tensor": np.array([[5, 6], [7, 8]], dtype=np.int32)} # int32
    print(dist_8_rev(arg1, arg2)) 
    # Positive
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.int64)} # int64
    arg2 = {"other_tensor": np.array([[5, 6], [7, 8]], dtype=np.int64)} # int64
    print(dist_8_rev(arg1, arg2))    
    # Negative (must be identical)
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.int64)} # int64
    arg2 = {"other_tensor": np.array([[5, 6], [7, 8]], dtype=np.int32)} # int32
    print(dist_8_rev(arg1, arg2))         
    # Negative
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.complex128)} # float, not int
    arg2 = {"other_tensor": np.array([[5, 6], [7, 8]], dtype=bool)} # int32  
    # Negative
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.int8)} # float, not int
    arg2 = {"other_tensor": np.array([[5, 6], [7, 8]], dtype=bool)} # int32        
    print(dist_8_rev(arg1, arg2)) # should be 0
    # Negative
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=bool)} # float, not int
    arg2 = {"other_tensor": np.array([[5, 6], [7, 8]], dtype=np.int8)} # int32        
    print(dist_8_rev(arg1, arg2)) # should be 0

if __name__ == "__main__":
    test_dist_8()