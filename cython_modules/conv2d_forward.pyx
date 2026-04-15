# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as cnp
import cython
from cython.parallel cimport prange

ctypedef cnp.float32_t f32

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline f32 _dot_kernel_3x3(
    f32[:, :, :, :] x_view,
    f32[:, :, :, :] k_view,
    Py_ssize_t b,
    Py_ssize_t oc,
    Py_ssize_t r,
    Py_ssize_t c,
    Py_ssize_t IC
) noexcept nogil:
    cdef Py_ssize_t ic
    cdef f32 acc = 0.0
    for ic in range(IC):
        acc += x_view[b, ic, r, c] * k_view[oc, ic, 0, 0]
        acc += x_view[b, ic, r, c + 1] * k_view[oc, ic, 0, 1]
        acc += x_view[b, ic, r, c + 2] * k_view[oc, ic, 0, 2]
        acc += x_view[b, ic, r + 1, c] * k_view[oc, ic, 1, 0]
        acc += x_view[b, ic, r + 1, c + 1] * k_view[oc, ic, 1, 1]
        acc += x_view[b, ic, r + 1, c + 2] * k_view[oc, ic, 1, 2]
        acc += x_view[b, ic, r + 2, c] * k_view[oc, ic, 2, 0]
        acc += x_view[b, ic, r + 2, c + 1] * k_view[oc, ic, 2, 1]
        acc += x_view[b, ic, r + 2, c + 2] * k_view[oc, ic, 2, 2]
    return acc

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline f32 _dot_kernel_generic(
    f32[:, :, :, :] x_view,
    f32[:, :, :, :] k_view,
    Py_ssize_t b,
    Py_ssize_t oc,
    Py_ssize_t r,
    Py_ssize_t c,
    Py_ssize_t IC,
    Py_ssize_t KH,
    Py_ssize_t KW
) noexcept nogil:
    cdef Py_ssize_t ic, kh, kw
    cdef f32 acc = 0.0
    for ic in range(IC):
        for kh in range(KH):
            for kw in range(KW):
                acc += x_view[b, ic, r + kh, c + kw] * k_view[oc, ic, kh, kw]
    return acc


def conv2d_forward_fused_cython(cnp.ndarray[f32, ndim=4] x,
                                cnp.ndarray[f32, ndim=4] kernels,
                                cnp.ndarray[f32, ndim=1] biases,
                                int stride,
                                int padding):
    cdef cnp.ndarray[f32, ndim=4] x_padded
    cdef cnp.ndarray[f32, ndim=4] out
    cdef f32[:, :, :, :] x_view
    cdef f32[:, :, :, :] k_view
    cdef f32[:] b_view
    cdef f32[:, :, :, :] out_view

    cdef Py_ssize_t B, IC, H, W
    cdef Py_ssize_t OC, KH, KW
    cdef Py_ssize_t OH, OW
    cdef Py_ssize_t b, oc, ic, oh, ow, kh, kw
    cdef Py_ssize_t r, c
    cdef Py_ssize_t total, bo
    cdef f32 bias_oc

    B = x.shape[0]
    IC = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]

    OC = kernels.shape[0]
    KH = kernels.shape[2]
    KW = kernels.shape[3]

    if padding > 0:
        x_padded = np.pad(
            x,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant'
        )
    else:
        x_padded = x

    OH = (x_padded.shape[2] - KH) // stride + 1
    OW = (x_padded.shape[3] - KW) // stride + 1
    out = np.zeros((B, OC, OH, OW), dtype=np.float32)

    x_view = x_padded
    k_view = kernels
    b_view = biases
    out_view = out

    total = B * OC
    if KH == 3 and KW == 3:
        for bo in prange(total, nogil=True, schedule='static'):
            b = bo // OC
            oc = bo - (b * OC)
            bias_oc = b_view[oc]
            for oh in range(OH):
                r = oh * stride
                for ow in range(OW):
                    c = ow * stride
                    out_view[b, oc, oh, ow] = _dot_kernel_3x3(
                        x_view, k_view, b, oc, r, c, IC
                    ) + bias_oc
    else:
        for bo in prange(total, nogil=True, schedule='static'):
            b = bo // OC
            oc = bo - (b * OC)
            bias_oc = b_view[oc]
            for oh in range(OH):
                r = oh * stride
                for ow in range(OW):
                    c = ow * stride
                    out_view[b, oc, oh, ow] = _dot_kernel_generic(
                        x_view, k_view, b, oc, r, c, IC, KH, KW
                    ) + bias_oc

    return out
