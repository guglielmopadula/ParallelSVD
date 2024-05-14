# Parallel SVD
This repo implements a Parallel SVD Algorithm (https://arxiv.org/pdf/1601.07010.pdf).
The matrix used for testing is generated via 

```
import numpy as np
np.random.seed(0)
A=np.random.rand(5,3)@np.random.rand(3,3)@np.random.rand(3,5)@np.random.rand(5,100000)
u,s,v=np.linalg.svd(A,full_matrices=False)
I=np.eye(5)
I[3,3]=0
I[4,4]=0
A=u@I@v

for i in range(10):
    np.save("tests/A_{}".format(i),A[:,i*10000:(i+1)*10000])

```