import ctypes as ct
import numba as nb
from numba import njit, types
import numpy as np
import os

quadpack_sig = types.double(types.double,
                            types.CPointer(types.double))

rootdir = os.path.dirname(os.path.realpath(__file__))+'/'
libquadpack = ct.CDLL(rootdir+'libcquadpack.so')

# # # # dqags
dqags_ = libquadpack.dqags
dqags_.argtypes = [ct.c_void_p, ct.c_double, ct.c_double, ct.c_double, ct.c_double, \
                 ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,]
dqags_.restype = ct.c_double
@njit
def dqags(funcptr, a, b, data = np.array([0.0], np.float64), epsabs = 1.49e-08, epsrel = 1.49e-08):
    abserr = np.array(0.0,np.float64)
    neval = np.array(0,np.int32)
    ier = np.array(0,np.int32)
    
    sol = dqags_(funcptr, a, b, epsabs, epsrel, \
                 abserr.ctypes.data, neval.ctypes.data, \
                 ier.ctypes.data, data.ctypes.data)
    
    success = True
    if ier != 0:
        success = False
        
    return sol, abserr.item(), success

# # # # dqagi
dqagi_ = libquadpack.dqagi
dqagi_.argtypes = [ct.c_void_p, ct.c_double, ct.c_void_p, ct.c_double, ct.c_double, \
                 ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,]
dqagi_.restype = ct.c_double
@njit
def dqagi(funcptr, a, b, data = np.array([0.0], np.float64), epsabs = 1.49e-08, epsrel = 1.49e-08):
    '''
    a - optional finite bound on integral.

    b - specifies range of integration as follows:
        b = -1 -- range is from -infinity to bound,
        b =  1 -- range is from bound to +infinity,
        b =  2 -- range is from -infinity to +infinity,
                (bound is immaterial in this case).
    '''
    abserr = np.array(0.0,np.float64)
    neval = np.array(0,np.int32)
    ier = np.array(0,np.int32)
    
    sol = dqagi_(funcptr, a, b, epsabs, epsrel, \
                 abserr.ctypes.data, neval.ctypes.data, \
                 ier.ctypes.data, data.ctypes.data)
    
    success = True
    if ier != 0:
        success = False
        
    return sol, abserr.item(), success

