import cv2
import numpy as np

def fourier_transform(input_img):
    img = input_img 
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fourier = np.fft.fft2(gray_img)
    fourier_shifted = np.fft.fftshift(fourier)
    return fourier_shifted

def low_high_filters(freq, img_data):
    rows_num, columns_num = img_data.shape
    lowpass_filter = np.zeros((rows_num, columns_num), dtype=np.float32)
    cutoff_freq = freq
    for i in range(rows_num):
        for j in range(columns_num):
            D = np.sqrt((i - rows_num / 2) ** 2 + (j - columns_num / 2) ** 2)
            if D <= cutoff_freq:
                lowpass_filter[i, j] = 1
            else:
                lowpass_filter[i, j] = 0
    highpass_filter = 1 - lowpass_filter
    return lowpass_filter, highpass_filter


def hybrid_image(hybrid_input_1, hybrid_input_2, freq_1, freq_2, combo_3_index, combo_4_index):
    w_1 , h_1 ,_ = hybrid_input_1.shape
    w_2 , h_2 , _ = hybrid_input_2.shape
    if w_1*h_1 > w_2*h_2:
        hybrid_input_1 = cv2.resize(hybrid_input_1, (w_2, h_2))
    else:
        hybrid_input_2 = cv2.resize(hybrid_input_2, (w_1, h_1))

    img_1 = fourier_transform(hybrid_input_1)
    img_2 = fourier_transform(hybrid_input_2)
    lowpass_filter_1, highpass_filter_1 = low_high_filters(float(freq_1), img_1)
    lowpass_filter_2, highpass_filter_2 = low_high_filters(float(freq_2), img_2)

    if combo_3_index == 1:
        img_1 = img_1 * lowpass_filter_1
    elif combo_3_index == 2:
        img_1 = img_1 * highpass_filter_1

    if combo_4_index == 1:
        img_2 = img_2 * lowpass_filter_2
    elif combo_4_index == 2:
        img_2 = img_2 * highpass_filter_2

    if img_1.shape != img_2.shape:
        print("Error: Images have different shapes")

    hybrid = img_1 + img_2
    # hybrid_img = np.fft.ifftshift(hybrid)
    hybrid_img_out =np.abs( np.fft.ifft2(hybrid)).astype(np.uint8)

    # hybrid_img_out = np.abs(np.fft.ifft2(hybrid_img))
    hybrid_img_out = cv2.cvtColor(hybrid_img_out, cv2.COLOR_GRAY2RGB)

    return hybrid_img_out
