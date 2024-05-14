import numpy as np
from mpi4py import MPI
from parallelsvd import compute_svd

def test_compute_us_rand():
    comm=MPI.COMM_WORLD
    rank=comm.rank
    matrix=np.load('tests/A_{}.npy'.format(rank))
    u,s,v=compute_svd(matrix,3,3)
    assert np.linalg.norm(u@np.diag(s)@v.T-matrix)/np.linalg.norm(matrix)<1e-04


def test_compute_us_full():
    comm=MPI.COMM_WORLD 
    rank=comm.rank
    matrix=np.load('tests/A_{}.npy'.format(rank))
    u,s,v=compute_svd(matrix,3,3,method="full")
    assert np.linalg.norm(u@np.diag(s)@v.T-matrix)/np.linalg.norm(matrix)<1e-04



def test_compute_us_single():
    comm=MPI.COMM_WORLD 
    rank=comm.rank
    matrix=np.load('tests/A_{}.npy'.format(rank))
    u,s,v=compute_svd(matrix,3,3,precision="single")
    assert np.linalg.norm(u@np.diag(s)@v.T-matrix)/np.linalg.norm(matrix)<1e-04


def test_compute_us_full_single():
    comm=MPI.COMM_WORLD
    rank=comm.rank
    matrix=np.load('tests/A_{}.npy'.format(rank))
    u,s,v=compute_svd(matrix,3,3,precision="single",method="full")
    assert np.linalg.norm(u@np.diag(s)@v.T-matrix)/np.linalg.norm(matrix)<1e-04
