import numpy as np
import adi
import cv2
from scipy.signal import butter, lfilter
import time

# ================== Настройки SDR ==================
video_bandwidth = 4e6       # Ширина полосы для видео
center_freq = 5865e6        # Центральная частота (5.865 ГГц)
sample_rate = 10e6          # Частота дискретизации
buffer_size = int(sample_rate * 0.04)  # ~40 мс (1 кадр PAL)

# Инициализация PlutoSDR
sdr = adi.Pluto("ip:192.168.2.1")
sdr.rx_lo = int(center_freq)
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(video_bandwidth * 1.2)
sdr.rx_buffer_size = buffer_size
sdr.gain_control_mode = 'manual'
sdr.rx_hardwaregain_chan0 = 20  # Усиление входного сигнала

# ================== Параметры видео PAL ==================
VISIBLE_LINES = 576         # Видимые строки
TOTAL_LINES = 625           # Всего строк в кадре
LINE_DURATION_US = 64       # Длительность строки в микросекундах
SAMPLES_PER_LINE = int(sample_rate * LINE_DURATION_US / 1e6)  # ≈ 640

WIDTH = 768
HEIGHT = VISIBLE_LINES

# ================== Пороги для синхроимпульсов ==================
VSYNC_THRESHOLD = 45        # Порог для VSYNC (уровень чёрного)
VSYNC_MIN_WIDTH = 100       # Минимальная длительность VSYNC (в отсчётах)
HSYNC_THRESHOLD = 44        # Порог для HSYNC
HSYNC_MIN_WIDTH = 15        # Минимальная ширина HSYNC
HSYNC_MIN_DISTANCE = 500    # Минимальное расстояние между HSYNC

# ================== Фильтр ==================
def butter_low(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

b, a = butter_low(video_bandwidth, sample_rate)

# ================== Поиск VSYNC ==================
def find_vsync(signal, threshold, min_width):
    vsync_start = -1
    vsync_end = -1
    
    for i in range(len(signal)):
        if signal[i] < threshold:
            if vsync_start == -1:
                vsync_start = i
            vsync_end = i
        else:
            if vsync_start != -1 and (vsync_end - vsync_start) >= min_width:
                return vsync_start, vsync_end
            vsync_start = -1
    return None, None

# ================== Поиск HSYNC ==================
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

# ================== Основной цикл ==================
frame_buffer = np.zeros((VISIBLE_LINES, WIDTH), dtype=np.uint8)
field_index = 0  # 0 или 1 для чересстрочности

try:
    while True:
        # Получение данных
        data = sdr.rx()
        
        # FM-демодуляция
        phase = np.unwrap(np.angle(data))
        demodulated = np.diff(phase) / (2 * np.pi * (1/sample_rate))
        
        # Фильтрация
        filtered = lfilter(b, a, demodulated)
        
        # Нормализация
        frame = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Поиск VSYNC
        vsync_start, vsync_end = find_vsync(frame, VSYNC_THRESHOLD, VSYNC_MIN_WIDTH)
        
        if vsync_start is not None:
            print(f"Найден VSYNC: {vsync_start}..{vsync_end}")
            
            # Область после VSYNC (первое поле)
            field_data = frame[vsync_end:vsync_end + SAMPLES_PER_LINE * TOTAL_LINES // 2]
            
            # Поиск HSYNC в поле
            hsync_pos = find_hsync_positions(field_data, HSYNC_THRESHOLD, HSYNC_MIN_WIDTH, HSYNC_MIN_DISTANCE)
            
            if len(hsync_pos) > 10:  # Если найдено достаточное количество HSYNC
                avg_line_length = int(np.median(np.diff(hsync_pos)))
                print(f"Средняя длина строки: {avg_line_length}")
                
                # Извлечение строк
                lines = []
                for i in range(len(hsync_pos) - 1):
                    start = hsync_pos[i]
                    end = start + avg_line_length
                    if end >= len(field_data):
                        break
                    line = field_data[start:end]
                    lines.append(line)
                
                # Очистка буфера перед заполнением нового поля
                if field_index == 0:
                    frame_buffer.fill(0)
                
                # Формирование кадра (чересстрочно)
                for i in range(min(len(lines), VISIBLE_LINES // 2)):
                    line_idx = i * 2 + field_index
                    if line_idx < VISIBLE_LINES:
                        line = lines[i]
                        resized_line = cv2.resize(line.reshape(1, -1), (WIDTH, 1))
                        frame_buffer[line_idx] = resized_line
                
                # Отображение
                display_img = cv2.medianBlur(frame_buffer, 3)
                cv2.imshow('PAL Video (VSYNC+HSYNC Sync)', display_img)
                
                # Переключение поля
                field_index ^= 1
                
                if cv2.waitKey(1) == ord('q'):
                    break

except KeyboardInterrupt:
    print("Прервано пользователем.")

finally:
    sdr.rx_destroy_buffer()
    cv2.destroyAllWindows()

    