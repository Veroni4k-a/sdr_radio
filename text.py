import numpy as np
import adi
import cv2
from scipy.signal import butter, lfilter

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

# Новые параметры для синхронизации
SYNC_THRESHOLD = 50  # Порог для обнаружения синхроимпульса (эмпирически)
MIN_SYNC_WIDTH = 20  # Минимальная длительность синхроимпульса в выборках
LINE_LENGTH = 768    # Ожидаемая длина строки в выборках (для 768 пикселей)

def find_hsync_positions(signal, threshold, min_width):
    """Находит позиции окончания горизонтальных синхроимпульсов"""
    #signal -array-voltage
    #threshold - low voltage ->sync
    #min_width - min width sync
    in_sync = False#located in sync (default False)
    sync_start = 0#index count sync(initially)
    syncs = []#array - end sync
    

    #cycle by signal
    #i - index
    #val -voltage
    for i, val in enumerate(signal):
        if val < threshold:
            if not in_sync:
                sync_start = i
                in_sync = True
        else:
            if in_sync:
                if (i - sync_start) >= min_width:
                    syncs.append(i)  # Запоминаем позицию конца синхроимпульса
                in_sync = False
    return syncs


def butter_low(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

b, a = butter_low(video_bandwidth, sample_rate)  # 4 МГц при 10 МГц Fs

try:
    while True:
        data = sdr.rx()
        print(data)
        
        # FM-демодуляция
        phase = np.unwrap(np.angle(data))
        demodulated = np.diff(phase) / (2 * np.pi * (1/sample_rate))
        print(demodulated)
        
        # Фильтрация
        filtered = lfilter(b, a, demodulated)
        print(filtered)
        
        # Нормализация
        frame = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        print(frame)
        
        # Поиск синхроимпульсов
        sync_positions = find_hsync_positions(frame, SYNC_THRESHOLD, MIN_SYNC_WIDTH)
        print(sync_positions)

        if len(sync_positions) < 2:
            continue  # Не нашли достаточно синхроимпульсов
        
        # Рассчет средней длины строки
        line_lengths = np.diff(sync_positions)
        avg_line_length = int(np.median(line_lengths))
        
        # Формируем кадр из полных строк
        lines = []
        for start in sync_positions[:-1]:
            end = start + LINE_LENGTH
            if end < len(frame):
                line = frame[start:end]
                lines.append(line)
        
        # Собираем изображение только если получили достаточно строк
        if len(lines) >= height:
            # Обрезаем до нужного количества строк
            image = np.array(lines[:height], dtype=np.uint8)
            
            # Обработка изображения
            image = cv2.medianBlur(image, 3)
            image = cv2.equalizeHist(image)
            
            # Масштабирование для отображения
            display_img = cv2.resize(image, (width, height))
            cv2.imshow('Analog Video', display_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    sdr.rx_destroy_buffer()
    cv2.destroyAllWindows()

"""Уже почти находит ширину кадра но есть маленькая погрешность с правой стороны, то есть ширина кадра + маленькая чать кадра другого,где это ошибка при поиске hsync.Может стоит учитывать"""