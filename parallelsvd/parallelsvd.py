import numpy as np
from mpi4py import MPI
import time
def _randomized_svd(A, rank):
    n = A.shape[1]
    P = np.random.randn(n, rank).astype(A.dtype)
    Z = A @ P
    Q, _ = np.linalg.qr(Z)
    Y = Q.T @ A
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    U = Q @ U
    return U, s, Vh

def _full_svd(A, rank):
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    return np.ascontiguousarray(U[:,:rank]), s[:rank], Vh[:rank]

def _svd(A, rank,method):
    if method=="randomized":
        U,s,_=_randomized_svd(A,rank)
        
    if method=="full":
        U,s,_=_full_svd(A,rank)
    return U,s

def _get_senders(node, current_level,acc,nprocs):
    arr = np.arange(node + acc**(current_level), node+acc**(current_level + 1), acc**(current_level))
    arr = arr[arr <= nprocs - 1]
    return arr

def _get_receiver(node, current_level,acc):
    return (acc**(current_level + 1)) * int(np.floor(node / (acc**(current_level + 1))))


def compute_svd(matrix,acc,matrix_rank,debug=False,method='randomized',precision='double'):
    '''
    Function to compute a parallel SVD. It assumes that the matrix in decomposed in column-block and that there is one of them in every rank. 

    :ndarray matrix: the matrix block.
    :int acc: the max number of messages that a process receives per iteration.
    :int matrix_rank: the SVD truncation rank
    :string debug: prints the messages between mpi ranks
    :string method: if randomized uses randomized svd, if full uses standard svd. Default is randomized.
    :precision: if single casts everything to float3d, if double casts everything to float64. Default is double.

    :return: U,s,V from of the truncated SVD.
    '''
    assert method=="randomized" or method=="full"
    assert precision=="double" or precision=="single"
    if precision=="single":
        dtype=np.float32
        mpi_dtype=MPI.FLOAT

    if precision=="double":
        dtype=np.float64
        mpi_dtype=MPI.DOUBLE


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    tot_num_steps = int(np.ceil(np.log(nprocs) / np.log(acc)))
    num_rows=matrix.shape[0]
    matrix=matrix.astype(dtype,copy=False)
    U,s=_svd(matrix,matrix_rank,method)
    matrix_tmp=U@np.diag(s)

    for i in range(tot_num_steps):
        if rank % (acc**(i + 1)) == 0:
            arr = _get_senders(rank, i,acc,nprocs)
            for a in arr:
                data = np.empty((num_rows,matrix_rank), dtype=dtype )
                print("Preparing to receive", a, rank, i)
                comm.Recv([data, mpi_dtype], source=a, tag=int(str(a) + str(rank)))
                matrix_tmp=np.concatenate((matrix_tmp,data),axis=1)
                if debug:
                    print("received", a, rank, i)
                    time.sleep(2)
            U,s=_svd(matrix_tmp,matrix_rank,method)
            matrix_tmp=U@np.diag(s)
        if rank % (acc**(i + 1)) != 0:
            if i==0 or rank % (acc**i)==0: 
                destination = _get_receiver(rank, i, acc)
                print("Preparing to send", rank, destination, i)
                comm.Send([matrix_tmp, mpi_dtype], dest=destination, tag=int(str(rank) + str(destination)))
                if debug:
                    print("sended", rank, destination, i)
                    time.sleep(2)

    

    if rank!=0:
        U=np.empty((matrix.shape[0],matrix_rank),dtype=dtype)
        s=np.empty((matrix_rank,),dtype=dtype)

    comm.Bcast([U,mpi_dtype],root=0)
    comm.Bcast([s,mpi_dtype],root=0)
    V=1/s*((matrix.T)@U)
    return U,s,V


