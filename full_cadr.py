"""Меняю proverka.py
итогом стало то что без дрожания стало"""
import numpy as np
import adi
import cv2
from scipy.signal import butter, lfilter

video_bandwidth = 4.8e6       # Ширина полосы для видео
center_freq = 5865e6        # Центральная частота (5.865 ГГц)
sample_rate = 10e6          # Частота дискретизации
buffer_size = int(sample_rate * 0.04)  # ~40 мс (1 кадр PAL)

sdr = adi.Pluto("ip:192.168.2.1")
sdr.rx_lo = int(center_freq)
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(video_bandwidth * 1.2)
sdr.rx_buffer_size = buffer_size
sdr.gain_control_mode = 'manual'
sdr.rx_hardwaregain_chan0 = 20  # Усиление входного сигнала


VISIBLE_LINES = 576         # Видимые строки
TOTAL_LINES = 625           # Всего строк в кадре
LINE_DURATION_US = 64       # Длительность строки в микросекундах
SAMPLES_PER_LINE = int(sample_rate * LINE_DURATION_US / 1e6)  # ≈ 640

WIDTH = 768
HEIGHT = VISIBLE_LINES

VSYNC_THRESHOLD = 35        # Порог для VSYNC (уровень чёрного)
VSYNC_MIN_WIDTH = 120       # Минимальная длительность VSYNC (в отсчётах)
HSYNC_THRESHOLD = 44        # Порог для HSYNC
HSYNC_MIN_WIDTH = 15        # Минимальная ширина HSYNC
HSYNC_MIN_DISTANCE = 600    # Минимальное расстояние между HSYNC

def butter_low(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

b, a = butter_low(video_bandwidth, sample_rate)

def find_vsync(signal, threshold, min_width):
    vsync_start = -1
    for i in range(len(signal)):
        if signal[i] < threshold:
            if vsync_start == -1:
                vsync_start = i
        else:
            if vsync_start != -1 and i - vsync_start >= min_width:
                return i  # Возвращаем конец VSYNC
            vsync_start = -1
    return None

def find_hsync_positions(signal, threshold, min_width, min_distance):
    syncs = []
    last_sync = -np.inf

    for i in range(len(signal)):
        if signal[i] < threshold:
            if i - last_sync > min_distance:
                start = i
                while i + 1 < len(signal) and signal[i + 1] < threshold:
                    i += 1
                if i - start >= min_width:
                    syncs.append(i)  # конец импульса
                    last_sync = i
    return syncs

frame_buffer = np.zeros((VISIBLE_LINES, WIDTH), dtype=np.uint8)
field_index = 0
fields_received = 0  # Считаем, сколько полей собрано

cv2.namedWindow('PAL Video Stream', cv2.WINDOW_AUTOSIZE)

try:
    while True:
        data = sdr.rx()

        phase = np.unwrap(np.angle(data))
        demodulated = np.diff(phase) / (2 * np.pi * (1/sample_rate))

        filtered = lfilter(b, a, demodulated)

        frame = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        vsync_pos = find_vsync(frame, VSYNC_THRESHOLD, VSYNC_MIN_WIDTH)
        if vsync_pos is not None:
            print(f"Найден VSYNC на позиции {vsync_pos}")
            field_data = frame[vsync_pos:]
            hsync_pos = find_hsync_positions(field_data, HSYNC_THRESHOLD, HSYNC_MIN_WIDTH, HSYNC_MIN_DISTANCE)

            if len(hsync_pos) > 10:
                avg_line_length = int(np.median(np.diff(hsync_pos)))
                lines = []

                for i in range(len(hsync_pos) - 1):
                    start = hsync_pos[i]
                    end = start + avg_line_length
                    if end >= len(field_data):
                        break
                    line = field_data[start:end]
                    lines.append(line)

                for i in range(min(len(lines), VISIBLE_LINES // 2)):
                    line_idx = i * 2 + field_index
                    if line_idx < VISIBLE_LINES:
                        resized_line = cv2.resize(lines[i].reshape(1, -1), (WIDTH, 1))
                        frame_buffer[line_idx] = resized_line

                fields_received += 1

                if fields_received >= 2:
                    display_img = cv2.medianBlur(frame_buffer, 3)
                    cv2.imshow('PAL Video Stream', display_img)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break

                    frame_buffer.fill(0)
                    fields_received = 0

                field_index ^= 1

except KeyboardInterrupt:
    print("Прервано пользователем.")

finally:
    sdr.rx_destroy_buffer()
    cv2.destroyAllWindows()


    """
    
    одно поле - 20 мкс
    один кадр - 40 мкс
    частота кадров 25 кадов

    
    


    """













