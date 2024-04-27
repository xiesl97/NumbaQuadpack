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

# # # # dqng
dqng_ = libquadpack.dqng
dqng_.argtypes = [ct.c_void_p, ct.c_double, ct.c_double, ct.c_double, ct.c_double, \
                 ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,]
dqng_.restype = ct.c_double
@njit
def dqng(funcptr, a, b, data = np.array([0.0], np.float64), epsabs = 1.49e-08, epsrel = 1.49e-08):
    abserr = np.array(0.0,np.float64)
    neval = np.array(0,np.int32)
    ier = np.array(0,np.int32)
    
    sol = dqng_(funcptr, a, b, epsabs, epsrel, \
                 abserr.ctypes.data, neval.ctypes.data, \
                 ier.ctypes.data, data.ctypes.data)
    
    success = True
    if ier != 0:
        success = False
        
    return sol, abserr.item(), success


# # # # dqag
dqag_ = libquadpack.dqag
dqag_.argtypes = [ct.c_void_p, ct.c_double, ct.c_double, ct.c_double, ct.c_double, \
                 ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,]
dqag_.restype = ct.c_double
@njit
def dqag(funcptr, a, b, data = np.array([0.0], np.float64), epsabs = 1.49e-08, epsrel = 1.49e-08, irule=1):
    '''
    irule - integration rule to be used as follows:
        irule = 1 -- G_K 7-15
        irule = 2 -- G_K 10-21
        irule = 3 -- G_K 15-31
        irule = 4 -- G_K 20-41
        irule = 5 -- G_K 25-51
        irule = 6 -- G_K 30-61
    '''
    abserr = np.array(0.0,np.float64)
    neval = np.array(0,np.int32)
    ier = np.array(0,np.int32)
    
    sol = dqag_(funcptr, a, b, epsabs, epsrel, irule, \
                 abserr.ctypes.data, neval.ctypes.data, \
                 ier.ctypes.data, data.ctypes.data)
    
    success = True
    if ier != 0:
        success = False
        
    return sol, abserr.item(), success

