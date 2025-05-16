import numpy as np
import adi
import cv2
from scipy.signal import butter, lfilter

# ================== Настройки SDR ==================
video_bandwidth = 4e6
center_freq = 5865e6
sample_rate = 10e6
buffer_size = 854 * 480 * 2  # Буфер на ~2 кадра

sdr = adi.Pluto("ip:192.168.2.1")
sdr.rx_lo = int(center_freq)
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(video_bandwidth * 1.2)
sdr.rx_buffer_size = buffer_size
sdr.gain_control_mode = 'manual'
sdr.rx_hardwaregain_chan0 = 20  # Можно повысить до 40–50 при слабом сигнале

# ================== Параметры видео ==================
height = 576  # Высота кадра PAL
width = 768   # Ширина кадра PAL
LINE_LENGTH = 768  # Ожидаемая длина строки (в отсчётах)

# ================== Пороги ==================
SYNC_THRESHOLD = 50      # Экспериментируй здесь!
MIN_HSYNC_WIDTH = 20     # Минимальная ширина HSYNC
MIN_VSYNC_LINES = 5      # Число аномальных строк подряд для VSYNC
MAX_LINE_DEVIATION = 0.3  # Отклонение длины строки для обнаружения VSYNC

# ================== Фильтр ==================
def butter_low(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

b, a = butter_low(video_bandwidth, sample_rate)

# ================== Поиск HSYNC ==================
def find_hsync_positions(signal, threshold, min_width):
    in_sync = False
    sync_start = 0
    syncs = []

    for i, val in enumerate(signal):
        if val < threshold:
            if not in_sync:
                sync_start = i
                in_sync = True
        else:
            if in_sync:
                if (i - sync_start) >= min_width:
                    syncs.append(i)  # Конец импульса
                in_sync = False
    return syncs

# ================== Поиск VSYNC ==================
def find_vsync(hsync_pos, min_missing_lines=5, max_deviation=0.3):
    if len(hsync_pos) < min_missing_lines + 2:
        return None

    line_lengths = np.diff(hsync_pos)
    avg_length = np.median(line_lengths)

    outliers = []
    for i in range(len(line_lengths)):
        if abs(line_lengths[i] - avg_length) > max_deviation * avg_length:
            outliers.append(i)

    # Ищем последовательность аномальных строк
    run_start = None
    for i in range(len(outliers) - min_missing_lines + 1):
        consecutive = True
        for j in range(min_missing_lines):
            if outliers[i+j] != outliers[i] + j:
                consecutive = False
                break
        if consecutive:
            run_start = outliers[i]
            break

    if run_start is not None and avg_length > LINE_LENGTH * 0.7:
        return hsync_pos[run_start]
    return None

# ================== Основной цикл ==================
frame_count = 0
last_vsync_pos = 0

try:
    while True:
        print("\n--- Новый кадр ---")

        data = sdr.rx()
        print(f"Получено {len(data)} отсчётов данных")

        # FM-демодуляция
        phase = np.unwrap(np.angle(data))
        demodulated = np.diff(phase) / (2 * np.pi * (1/sample_rate))
        print(f"Демодулированных отсчётов: {len(demodulated)}")

        # Фильтрация
        filtered = lfilter(b, a, demodulated)
        print(f"Фильтрованных отсчётов: {len(filtered)}")

        # Нормализация
        frame = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        print(f"Нормализованных отсчётов: {len(frame)}")

        # Поиск HSYNC
        hsync_pos = find_hsync_positions(frame, SYNC_THRESHOLD, MIN_HSYNC_WIDTH)
        print(f"Найдено HSYNC: {len(hsync_pos)}")

        if len(hsync_pos) < 2:
            print("Не хватает HSYNC для анализа")
            continue

        # Поиск VSYNC
        vsync_start = find_vsync(hsync_pos, MIN_VSYNC_LINES, MAX_LINE_DEVIATION)
        print(f"VSYNC начат с позиции: {vsync_start}")

        if vsync_start is not None:
            last_vsync_pos = vsync_start

        # Берём только те HSYNC, что после последнего VSYNC
        current_hsync = [p for p in hsync_pos if p >= last_vsync_pos]
        print(f"HSYNC после VSYNC: {len(current_hsync)}")

        if len(current_hsync) >= height:
            try:
                line_length = int(np.median(np.diff(current_hsync)))
                print(f"Средняя длина строки: {line_length}")

                # Берём только первые height строк после VSYNC
                selected_hsync = current_hsync[:height]

                # Формируем кадр
                lines = []
                for start in selected_hsync:
                    end = start + line_length
                    if end < len(frame):
                        line = frame[start:end]
                        lines.append(line)

                print(f"Сформировано строк: {len(lines)}")

                if len(lines) == height:
                    image = np.array(lines, dtype=np.uint8)
                    print(f"Размер собранного кадра: {image.shape}")

                    # Обработка изображения
                    image = cv2.medianBlur(image, 3)
                    image = cv2.equalizeHist(image)

                    # Масштабируем и выводим
                    display_img = cv2.resize(image, (width, height))
                    cv2.imshow('Analog Video', display_img)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break

                    frame_count += 1
                    print(f"Кадр #{frame_count} успешно отображён")

                    # Сбрасываем позицию VSYNC после вывода кадра
                    last_vsync_pos = 0

            except Exception as e:
                print("Ошибка при формировании кадра:", e)

except KeyboardInterrupt:
    print("Прервано пользователем.")

finally:
    sdr.rx_destroy_buffer()
    cv2.destroyAllWindows()