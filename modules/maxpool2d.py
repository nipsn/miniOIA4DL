from modules.layer import Layer
#from cython_modules.maxpool2d import maxpool_forward_cython
import numpy as np

class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input, training=True):
        B, C, H, W = input.shape

        # Fast path for 2x2 kernel with stride 2 and even dimensions (OIAnet)
        if self.kernel_size == 2 and self.stride == 2 and H % 2 == 0 and W % 2 == 0:
            if training:
                return self._forward_fast_2x2_training(input)
            return self._forward_fast_2x2_inference(input)

        return self._forward_standard(input)
    
    def _forward_standard(self, input):
        self.input = input
        B, C, H, W = input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1

        self.max_indices = np.zeros((B, C, out_h, out_w, 2), dtype=int)
        output = np.zeros((B, C, out_h, out_w),dtype=input.dtype)

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * SH
                        h_end = h_start + KH
                        w_start = j * SW
                        w_end = w_start + KW

                        window = input[b, c, h_start:h_end, w_start:w_end]
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        max_val = window[max_idx]

                        output[b, c, i, j] = max_val
                        self.max_indices[b, c, i, j] = (h_start + max_idx[0], w_start + max_idx[1])

        return output

    def _forward_fast_2x2_training(self, input):
        self.input = input
        B, C, H, W = input.shape

        out_h = H // 2
        out_w = W // 2

        x00 = input[:, :, 0::2, 0::2]
        x01 = input[:, :, 0::2, 1::2]
        x10 = input[:, :, 1::2, 0::2]
        x11 = input[:, :, 1::2, 1::2]

        stacked = np.stack((x00, x01, x10, x11), axis=-1)   # (B, C, out_h, out_w, 4)

        output = np.max(stacked, axis=-1)
        arg = np.argmax(stacked, axis=-1)                   # valores 0,1,2,3

        row_off = arg // 2
        col_off = arg % 2

        base_rows = (2 * np.arange(out_h))[None, None, :, None]
        base_cols = (2 * np.arange(out_w))[None, None, None, :]

        self.max_indices = np.empty((B, C, out_h, out_w, 2), dtype=int)
        self.max_indices[..., 0] = base_rows + row_off
        self.max_indices[..., 1] = base_cols + col_off

        return output

    def _forward_fast_2x2_inference(self, input):
        x00 = input[:, :, 0::2, 0::2]
        x01 = input[:, :, 0::2, 1::2]
        x10 = input[:, :, 1::2, 0::2]
        x11 = input[:, :, 1::2, 1::2]
        return np.maximum(
            np.maximum(x00, x01),
            np.maximum(x10, x11)
        )

    def backward(self, grad_output, learning_rate=None):
        B, C, H, W = self.input.shape
        grad_input = np.zeros_like(self.input, dtype=grad_output.dtype)
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        r, s = self.max_indices[b, c, i, j]
                        grad_input[b, c, r, s] += grad_output[b, c, i, j]

        return grad_input
