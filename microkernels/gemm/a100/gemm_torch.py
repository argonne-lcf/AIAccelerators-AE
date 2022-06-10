import numpy as np
import torch
import time

device = torch.cuda.current_device()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

config = (
    #    m,    n,    k, batch_size, AT,    BT  (row order)
    (   64, 1760, 1760, 1, False, False),
    ( 2560,   64, 2560, 1, False, False),
    ( 1760,  128, 1760, 1, False, False),
    ( 2560, 2560, 2560, 1, False, False),

    (   64, 1760, 1760, 64, False, False),
    ( 2560,   64, 2560, 64, False, False),
    ( 1760,  128, 1760, 64, False, False),
    ( 2560, 2560, 2560, 64, False, False),

    (   64, 1760, 1760, 128, False, False),
    ( 2560,   64, 2560, 128, False, False),
    ( 1760,  128, 1760, 128, False, False),
    ( 2560, 2560, 2560, 128, False, False),

    (   64, 1760, 1760, 256, False, False),
    ( 2560,   64, 2560, 256, False, False),
    ( 1760,  128, 1760, 256, False, False),
    ( 2560, 2560, 2560, 256, False, False),
)

print("M,N,K,Batch_Size,Data_Type,TIME,TFLOPS")

for dataType in (torch.float32, torch.float16, torch.bfloat16): #np.float32, np.float16,

    for m, n, k, batch_size, at, bt in config:

        # initial dimentions not considering batch_size
        # dimA = (k,m) if at else (m,k)
        # dimB = (n,k) if bt else (k,n)
        # dimC = (m,n)

        opA = 'T' if at else 'N'
        opB = 'T' if bt else 'N'
        op  = opA + opB

        # A = torch.randn(dimA, dtype=dataType).to(device)
        # B = torch.randn(dimB, dtype=dataType).to(device)
        A= torch.randn(batch_size, m, n,dtype=dataType).to(device)
        B= torch.randn(batch_size, n, k,dtype=dataType).to(device)

        if at: A = A.t()
        if bt: B = B.t()

        C = torch.matmul(A, B)

        n_iters = 100
        start.record()
        for loop in range(n_iters):
            C = torch.matmul(A, B)
        end.record()
        torch.cuda.synchronize()

        tflops = (2*m*n*k*batch_size)/(start.elapsed_time(end)/n_iters)/10**9 #tflops

        print("%d,%d,%d,%d,%s,%f,%f" % (m, n, k,batch_size, A.dtype,start.elapsed_time(end)/n_iters/1000, tflops ))