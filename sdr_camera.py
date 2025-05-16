import numpy as np #for working with array
import adi#for working with API PLUTO
import cv2#for working Opencv
from scipy.signal import butter, lfilter #library filter 

""" Settings SDR"""
video_bandwidth = 4e6  #bandwidth video  
center_freq = 5865e6 #center freq
sample_rate = 10e6 # sample rate 
buffer_size = 854 * 480 * 2  # buffer for strong a frame

"""PlutoSDR init """
sdr = adi.Pluto("ip:192.168.2.1")
sdr.rx_lo = int(center_freq)
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(video_bandwidth * 1.2)
sdr.rx_buffer_size = buffer_size
sdr.gain_control_mode = 'manual'
sdr.rx_hardwaregain_chan0 = 20

"""Setting video"""
aspect_ratio = '4:3'
height = 576
width = 768

"""Filter low-freq"""
def butter_low(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

b, a = butter_low(video_bandwidth, sample_rate)  # 4 МГц при 10 МГц Fs

try:
    while True:
        data = sdr.rx()
        
        # FM-demodulation 
        if len(data) < 2:
            continue
        phase = np.unwrap(np.angle(data))
        demodulated = np.diff(phase) / (2 * np.pi * (1/sample_rate))
        
        # Filter
        filtered = lfilter(b, a, demodulated)
        
        # Normalization
        frame = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Create frame
        required_samples = width * height
        if len(frame) >= required_samples:
            image = frame[:required_samples].reshape((height, width))
            
            
            image = cv2.medianBlur(image, 3)#Salt and paper
            image = cv2.equalizeHist(image)#Contrast
            
            cv2.imshow('Analog Video', image)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    sdr.rx_destroy_buffer()
    cv2.destroyAllWindows()
"""Поправь код чтобы убрать дробление полученного кадра на несколько частей (вижу что камера работает но вместо 1 кадра получаю полосы а в этих полосах по несколько кадров )я думаю это происзодит изи за того что не ищу vsync  учитывай что стандарт PAL(пока ищи только vsync)"""
