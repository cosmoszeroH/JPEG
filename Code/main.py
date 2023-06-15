import numpy as np
import cv2
import scipy.fftpack as fftpack
import skimage.measure
from matplotlib import pyplot as plt


# dam bao nam trong khoang
def Clamp(col):
    col = 255 if col > 255 else col
    col = 0 if col < 0 else col
    return int(col)


# chia mot mang thanh nhieu mang con
def split(array, nrows, ncols):
    r, h = array.shape
    return (array.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


def multiply_matrix(matrix, shape):
    m = 0
    n = 0
    multi_matrix = np.zeros(shape)
    while m < shape[0]:
        while n < shape[1]:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    multi_matrix[m+i][n+j]=matrix[i][j]
            n += matrix.shape[1]
        m += matrix.shape[0]
        n = 0

    return multi_matrix


def encode_quant(orig, quant):
    return (orig / quant).astype(int)


def decode_quant(orig, quant):
    return (orig * quant).astype(int)


def encode_dct(orig):
    return fftpack.dctn(orig, norm='ortho')


def decode_dct(orig):
    return fftpack.idctn(orig, norm='ortho')


def encode_decode(orig, quant):
    enc_dct = encode_dct(orig)
    enc_quant = encode_quant(enc_dct, quant)
    dec = decode_quant(enc_quant, quant)
    dec = decode_dct(dec)
    return dec


def upsample_matrix(orig, ratio):
    upsample = np.zeros((orig.shape[0]*ratio, orig.shape[1]*ratio))

    m = 0
    n = 0
    i = 0
    j = 0
    while m < upsample.shape[0]:
        while n < upsample.shape[1]:
            for u in range(ratio):
                for v in range(ratio):
                    upsample[m+u][n+v] = orig[i][j]
            n += ratio
            j += 1
        m += ratio
        n = 0
        i += 1
        j = 0

    return upsample


"""mini_luminance_quantization_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])"""

mini_luminance_quantization_table = np.array([
    [4, 3, 4, 4, 4, 6, 11, 15],
    [3, 3, 3, 4, 5, 8, 14, 19],
    [3, 4, 4, 5, 8, 12, 16, 20],
    [4, 5, 6, 7, 12, 14, 18, 20],
    [6, 6, 9, 11, 14, 17, 21, 23],
    [9, 12, 12, 18, 23, 22, 25, 21],
    [11, 13, 15, 17, 21, 23, 25, 21],
    [13, 12, 12, 13, 16, 19, 21, 21]
])

"""mini_chrominance_quantization_table = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])"""

mini_chrominance_quantization_table = np.array([
    [4, 4, 6, 10, 21, 21, 21, 21],
    [4, 5, 6, 21, 21, 21, 21, 21],
    [6, 6, 12, 21, 21, 21, 21, 21],
    [10, 14, 21, 21, 21, 21, 21, 21],
    [21, 21, 21, 21, 21, 21, 21, 21],
    [21, 21, 21, 21, 21, 21, 21, 21],
    [21, 21, 21, 21, 21, 21, 21, 21],
    [21, 21, 21, 21, 21, 21, 21, 21]
])

ratio = 2

image = 'lena_rgb.png'
imageYCC = cv2.imread(image, cv2.COLOR_BGR2YCR_CB)

plt.imshow(imageYCC.astype(np.uint8))
plt.show()

cv2.imwrite('lena.jpeg', imageYCC)

a, b, c = imageYCC.shape  # kich thuoc cua anh
Y_array = np.zeros([a, b])
Cb_array = np.zeros([a, b])
Cr_array = np.zeros([a, b])

# chuyen khoang gia tri tu [0; 255] ve [-128; 127]
for m in range(a):
    for n in range(b):
        Y_array[m][n] = Clamp(imageYCC[m, n, 0]) - 128
        Cb_array[m][n] = Clamp(imageYCC[m, n, 2]) - 128
        Cr_array[m][n] = Clamp(imageYCC[m, n, 1]) - 128

luminance_quantization_table = multiply_matrix(mini_luminance_quantization_table, Y_array.shape)
Y_array_process = encode_decode(Y_array, luminance_quantization_table)

"""# giam kich thuoc cua Cb, Cr
Cb_array_downsample = skimage.measure.block_reduce(Cb_array, block_size=(ratio, ratio), func=np.mean, cval=0)
Cr_array_downsample = skimage.measure.block_reduce(Cr_array, block_size=(ratio, ratio), func=np.mean, cval=0)

chrominance_quantization_table = multiply_matrix(mini_chrominance_quantization_table, Cb_array_downsample.shape)

Cb_array_downsample_process = encode_decode(Cb_array_downsample, chrominance_quantization_table)
Cr_array_downsample_process = encode_decode(Cr_array_downsample, chrominance_quantization_table)

Cb_array_process = upsample_matrix(Cb_array_downsample, ratio)
Cr_array_process = upsample_matrix(Cr_array_downsample, ratio)"""

chrominance_quantization_table = multiply_matrix(mini_chrominance_quantization_table, Cb_array.shape)
Cb_array_process = encode_decode(Cb_array, chrominance_quantization_table)
Cr_array_process = encode_decode(Cr_array, chrominance_quantization_table)

# chuyen khoang gia tri tu [-128; 127] ve [0; 255]
for i in range(Y_array_process.shape[0]):
    for j in range(Y_array_process.shape[1]):
        Y_array_process[i][j] += 128
        Cb_array_process[i][j] += 128
        Cr_array_process[i][j] += 128

compress_image = np.zeros(imageYCC.shape, dtype=float)
for i in range(compress_image.shape[0]):
    for j in range(compress_image.shape[1]):
        compress_image[i][j][0] = Clamp(Y_array_process[i][j])
        compress_image[i][j][1] = Clamp(Cr_array_process[i][j])
        compress_image[i][j][2] = Clamp(Cb_array_process[i][j])

plt.imshow(compress_image.astype(np.uint8))
plt.show()

#cv2.imwrite('lena_compress.jpeg', compress_image)