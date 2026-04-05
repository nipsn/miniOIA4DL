import numpy as np
from modules.conv2d import Conv2D


def test_conv2d():
    # Parameters
    img_width = 5
    img_height = 5
    in_channels = 1
    out_channels = 3
    kernel_size = 3
    stride = 2
    padding = 1
    batch_size = 2
    conv_algo = 0  # Assuming conv_algo is not used in this test

    # Initialize layer
    conv = Conv2D(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size, stride=stride, padding=padding,conv_algo=conv_algo)
    
    # Input: 1 image, 1 channel, 5x5 values from 0 to 24
    input_image = np.arange(img_height*img_height*in_channels*batch_size, dtype=np.float32).reshape(batch_size, in_channels, img_width, img_height)

    # Set all kernels to 1 and biases to 0
    conv.kernels = np.ones((out_channels, in_channels, kernel_size, kernel_size), dtype=np.float32)
    conv.biases = np.zeros(out_channels, dtype=np.float32)

    # Pad the input manually for expected output
    padded = np.pad(input_image, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    # Compute expected output manually
    out_h = (padded.shape[2] - kernel_size) // stride + 1
    out_w = (padded.shape[3] - kernel_size) // stride + 1
    expected_output = np.zeros((batch_size, out_channels, out_h, out_w), dtype=np.float32)

    for b in range(batch_size):
        for c in range(out_channels):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    w_start = j * stride
                    patch = padded[b, 0, h_start:h_start+kernel_size, w_start:w_start+kernel_size]
                    expected_output[b, c, i, j] = np.sum(patch)  # kernel is all ones

    # Run the actual forward pass
    output = conv.forward(input_image)

    # Validate
    assert np.allclose(output, expected_output), "Conv2D (padding+stride) forward mismatch!"
    print("✅ Conv2D forward with padding and stride passed!")

test_conv2d()

def test_conv2d_im2col_simple():
    img_width = 5
    img_height = 5
    in_channels = 1
    out_channels = 3
    kernel_size = 3
    stride = 2
    padding = 1
    batch_size = 2

    conv = Conv2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        conv_algo=1,   # tu algoritmo nuevo
    )

    x = np.arange(
        img_height * img_width * in_channels * batch_size,
        dtype=np.float32
    ).reshape(batch_size, in_channels, img_width, img_height)

    conv.kernels = np.ones((out_channels, in_channels, kernel_size, kernel_size), dtype=np.float32)
    conv.biases = np.zeros(out_channels, dtype=np.float32)

    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")

    out_h = (padded.shape[2] - kernel_size) // stride + 1
    out_w = (padded.shape[3] - kernel_size) // stride + 1
    expected = np.zeros((batch_size, out_channels, out_h, out_w), dtype=np.float32)

    for b in range(batch_size):
        for c in range(out_channels):
            for i in range(out_h):
                for j in range(out_w):
                    h0 = i * stride
                    w0 = j * stride
                    patch = padded[b, 0, h0:h0+kernel_size, w0:w0+kernel_size]
                    expected[b, c, i, j] = np.sum(patch)

    out = conv.forward(x)

    print("shape esperada:", expected.shape)
    print("shape obtenida:", out.shape)

    assert out.shape == expected.shape, f"Shape incorrecta: {out.shape} != {expected.shape}"
    print("max abs diff:", np.max(np.abs(out - expected)))
    assert np.allclose(out, expected, atol=1e-6), "im2col no coincide con el caso manual"

test_conv2d_im2col_simple()


def compare_direct_vs_im2col():
    np.random.seed(0)

    in_channels = 3
    out_channels = 8
    kernel_size = 3
    stride = 1
    padding = 1
    batch_size = 4
    height = 16
    width = 16

    x = np.random.randn(batch_size, in_channels, height, width).astype(np.float32)

    conv_direct = Conv2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        conv_algo=0,
    )

    conv_im2col = Conv2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        conv_algo=1,
    )

    conv_im2col.kernels = conv_direct.kernels.copy()
    conv_im2col.biases = conv_direct.biases.copy()

    out_direct = conv_direct.forward(x)
    out_im2col = conv_im2col.forward(x)

    diff = np.abs(out_direct - out_im2col)

    print("shape direct :", out_direct.shape)
    print("shape im2col :", out_im2col.shape)
    print("max abs diff :", diff.max())
    print("mean abs diff:", diff.mean())

    assert out_direct.shape == out_im2col.shape
    assert np.allclose(out_direct, out_im2col, atol=1e-5, rtol=1e-5), \
        "La salida de im2col no coincide con direct"

compare_direct_vs_im2col()

def test_oianet_like_shapes():
    np.random.seed(7)

    shapes = [
        (8, 3, 32, 32, 32),
        (8, 32, 16, 16, 64),
        (8, 64, 8, 8, 128),
    ]

    for B, Cin, H, W, Cout in shapes:
        x = np.random.randn(B, Cin, H, W).astype(np.float32)

        conv0 = Conv2D(Cin, Cout, 3, stride=1, padding=1, conv_algo=0)
        conv1 = Conv2D(Cin, Cout, 3, stride=1, padding=1, conv_algo=1)

        conv1.kernels = conv0.kernels.copy()
        conv1.biases = conv0.biases.copy()

        y0 = conv0.forward(x)
        y1 = conv1.forward(x)

        diff = np.abs(y0 - y1)
        print(f"OIANet-like {x.shape} -> max diff {diff.max()} mean diff {diff.mean()}")

        assert np.allclose(y0, y1, atol=1e-5, rtol=1e-5)

test_oianet_like_shapes()


def test_conv2d_im2colfused_simple():
    img_width = 5
    img_height = 5
    in_channels = 1
    out_channels = 3
    kernel_size = 3
    stride = 2
    padding = 1
    batch_size = 2

    conv = Conv2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        conv_algo=2,
    )

    x = np.arange(
        img_height * img_width * in_channels * batch_size,
        dtype=np.float32
    ).reshape(batch_size, in_channels, img_width, img_height)

    conv.kernels = np.ones((out_channels, in_channels, kernel_size, kernel_size), dtype=np.float32)
    conv.biases = np.zeros(out_channels, dtype=np.float32)

    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")

    out_h = (padded.shape[2] - kernel_size) // stride + 1
    out_w = (padded.shape[3] - kernel_size) // stride + 1
    expected = np.zeros((batch_size, out_channels, out_h, out_w), dtype=np.float32)

    for b in range(batch_size):
        for c in range(out_channels):
            for i in range(out_h):
                for j in range(out_w):
                    h0 = i * stride
                    w0 = j * stride
                    patch = padded[b, 0, h0:h0+kernel_size, w0:w0+kernel_size]
                    expected[b, c, i, j] = np.sum(patch)

    out = conv.forward(x)

    print("shape esperada (fused):", expected.shape)
    print("shape obtenida (fused):", out.shape)
    print("max abs diff (fused):", np.max(np.abs(out - expected)))

    assert out.shape == expected.shape, f"Shape incorrecta: {out.shape} != {expected.shape}"
    assert np.allclose(out, expected, atol=1e-6), "im2colfused no coincide con el caso manual"


test_conv2d_im2colfused_simple()


def compare_direct_vs_im2colfused():
    np.random.seed(1)

    in_channels = 3
    out_channels = 8
    kernel_size = 3
    stride = 1
    padding = 1
    batch_size = 4
    height = 16
    width = 16

    x = np.random.randn(batch_size, in_channels, height, width).astype(np.float32)

    conv_direct = Conv2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        conv_algo=0,
    )

    conv_fused = Conv2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        conv_algo=2,
    )

    conv_fused.kernels = conv_direct.kernels.copy()
    conv_fused.biases = conv_direct.biases.copy()

    out_direct = conv_direct.forward(x)
    out_fused = conv_fused.forward(x)

    diff = np.abs(out_direct - out_fused)

    print("shape direct  :", out_direct.shape)
    print("shape fused   :", out_fused.shape)
    print("max abs diff  :", diff.max())
    print("mean abs diff :", diff.mean())

    assert out_direct.shape == out_fused.shape
    assert np.allclose(out_direct, out_fused, atol=1e-5, rtol=1e-5), \
        "La salida de im2colfused no coincide con direct"


compare_direct_vs_im2colfused()


def test_oianet_like_shapes_im2colfused():
    np.random.seed(9)

    shapes = [
        (8, 3, 32, 32, 32),
        (8, 32, 16, 16, 64),
        (8, 64, 8, 8, 128),
    ]

    for B, Cin, H, W, Cout in shapes:
        x = np.random.randn(B, Cin, H, W).astype(np.float32)

        conv0 = Conv2D(Cin, Cout, 3, stride=1, padding=1, conv_algo=0)
        conv2 = Conv2D(Cin, Cout, 3, stride=1, padding=1, conv_algo=2)

        conv2.kernels = conv0.kernels.copy()
        conv2.biases = conv0.biases.copy()

        y0 = conv0.forward(x)
        y2 = conv2.forward(x)

        diff = np.abs(y0 - y2)
        print(f"OIANet-like fused {x.shape} -> max diff {diff.max()} mean diff {diff.mean()}")

        assert np.allclose(y0, y2, atol=1e-5, rtol=1e-5)


test_oianet_like_shapes_im2colfused()
