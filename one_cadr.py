"""Принимает валидно один кадр"""
import numpy as np
import adi
import cv2
from scipy.signal import butter, lfilter
import time

# ================== Настройки SDR ==================
video_bandwidth = 4e6       # Ширина полосы для видео
center_freq = 5865e6        # Центральная частота (5.865 ГГц)
sample_rate = 10e6          # Частота дискретизации
buffer_size = int(sample_rate * 0.05)  # ~50 мс данных (~500 тыс. отсчётов)

# Инициализация PlutoSDR
sdr = adi.Pluto("ip:192.168.2.1")
sdr.rx_lo = int(center_freq)
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(video_bandwidth * 1.2)
sdr.rx_buffer_size = buffer_size
sdr.gain_control_mode = 'manual'
sdr.rx_hardwaregain_chan0 = 20  # Усиление входного сигнала

# ================== Параметры видео PAL ==================
VISIBLE_LINES = 576     # Видимые строки
TOTAL_LINES = 625       # Всего строк в кадре
LINE_DURATION_US = 64   # Длительность строки в микросекундах
SAMPLES_PER_LINE = int(sample_rate * LINE_DURATION_US / 1e6)  # ≈ 640

WIDTH = 768
HEIGHT = VISIBLE_LINES

# ================== Пороги ==================
SYNC_THRESHOLD = 44  # Порог для поиска HSYNC
MIN_HSYNC_WIDTH = 15     # Минимальная ширина импульса HSYNC
MIN_HSYNC_DISTANCE = 500 # Минимальное расстояние между HSYNC

# ================== Фильтр ==================
def butter_low(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

b, a = butter_low(video_bandwidth, sample_rate)

# ================== Поиск HSYNC ==================
def find_hsync_positions(signal, threshold, min_width, min_distance=500):
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
lines = []  # Буфер для строк кадра
frame_count = 0
field_index = 0  # Для чересстрочной развёртки
last_frame_time = time.time()
TARGET_FPS = 25
FRAME_INTERVAL = 1.0 / TARGET_FPS  # ~0.04 секунды

try:
    while True:
        current_time = time.time()
        if current_time - last_frame_time < FRAME_INTERVAL:
            time.sleep(max(0, FRAME_INTERVAL - (current_time - last_frame_time)))
        last_frame_time = time.time()

        print("\n--- Получение новых данных ---")
        data = sdr.rx()
        print(f"Получено {len(data)} отсчётов")

        # FM-демодуляция
        phase = np.unwrap(np.angle(data))
        demodulated = np.diff(phase) / (2 * np.pi * (1/sample_rate))

        # Фильтрация
        filtered = lfilter(b, a, demodulated)

        # Нормализация
        frame = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Поиск HSYNC
        hsync_pos = find_hsync_positions(frame, SYNC_THRESHOLD, MIN_HSYNC_WIDTH, MIN_HSYNC_DISTANCE)
        print(f"Найдено HSYNC: {len(hsync_pos)}")

        if len(hsync_pos) < 2:
            print("Не хватает HSYNC для анализа")
            continue

        line_length = int(np.median(np.diff(hsync_pos)))
        print(f"Средняя длина строки: {line_length}")

        # Извлечение строк
        for idx, start in enumerate(hsync_pos):
            end = start + line_length
            if end >= len(frame):
                break
            line = frame[start:end]
            lines.append(line)

        # Если собрали достаточно строк для кадра
        if len(lines) >= TOTAL_LINES:
            full_frame = []
            for i in range(VISIBLE_LINES):
                # Выбор строк в зависимости от поля (чересстрочка)
                line_idx = i // 2 + field_index * (VISIBLE_LINES // 2)
                if line_idx < len(lines):
                    full_frame.append(lines[line_idx])
            image = np.array(full_frame[:VISIBLE_LINES], dtype=np.uint8)
            print(f"Размер собранного кадра: {image.shape}")

            # Обработка изображения
            image = cv2.medianBlur(image, 3)
            image = cv2.equalizeHist(image)

            # Избавляемся от одномерного канала
            if len(image.shape) == 3 and image.shape[2] == 1:
                image = image.reshape((image.shape[0], image.shape[1]))

            display_img = cv2.resize(image, (WIDTH, HEIGHT))
            cv2.imshow('PAL Video Fullscreen', display_img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            frame_count += 1
            print(f"✅ Кадр #{frame_count} успешно отображён")

            # Переключаем поле
            field_index ^= 1

            # Удаляем использованные строки
            lines = lines[VISIBLE_LINES:]

except KeyboardInterrupt:
    print("Прервано пользователем.")

finally:
    sdr.rx_destroy_buffer()
    cv2.destroyAllWindows()

    """Ловит кадр но, что происходит - он делит его на две части по горизонтали и вторая часть идет свверху,а вторая снизу
     при изменении порога погрешность эта меняется (меньше потери куска от кадра очень маленький),что мне сделать чтоюы был один кадр, без потери куска
     . так же я посмотрела настройки приемника и  поняла чтоу меня bandwidth равнен 4,8, может ли это давать ошибку?

     Так же пока я получаю кадр а хотелось бы видео в реальном времени
    
    """
