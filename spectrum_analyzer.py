import math
import struct
import matplotlib.pyplot as plt


# --- 1. ФУНКЦИИ ИЗ МЕТОДИЧКИ (Приложение 1, стр. 12-13) ---

def fft_manual(xk_re):
    """Прямое БПФ по методичке (возвращает Re и Im части спектра)"""
    N = len(xk_re)
    m = int(math.log2(N))
    xk_im = [0.0] * N

    # Блок инверсии бит
    j = 0
    for i in range(N - 1):
        if i < j:
            xk_re[i], xk_re[j] = xk_re[j], xk_re[i]
            xk_im[i], xk_im[j] = xk_im[j], xk_im[i]
        k = N // 2
        while k <= j:
            j -= k
            k //= 2
        j += k

    # Каскады БПФ
    for L in range(1, m + 1):
        le = 2 ** L
        le1 = le // 2
        u_re, u_im = 1.0, 0.0
        arg = math.pi / le1
        w_re = math.cos(arg)
        w_im = -math.sin(arg)  # Минус для прямого преобразования

        for j in range(le1):
            for i in range(j, N, le):
                ip = i + le1
                # Операция "Бабочка"
                t_re = xk_re[ip] * u_re - xk_im[ip] * u_im
                t_im = xk_re[ip] * u_im + xk_im[ip] * u_re
                xk_re[ip] = xk_re[i] - t_re
                xk_im[ip] = xk_im[i] - t_im
                xk_re[i] = xk_re[i] + t_re
                xk_im[i] = xk_im[i] + t_im
            temp_u_re = u_re * w_re - u_im * w_im
            u_im = u_re * w_im + u_im * w_re
            u_re = temp_u_re
    return xk_re, xk_im


def ifft_manual(re, im):
    """Обратное БПФ для восстановления сигнала из спектра"""
    N = len(re)
    m = int(math.log2(N))

    # Инверсия бит
    j = 0
    for i in range(N - 1):
        if i < j:
            re[i], re[j] = re[j], re[i]
            im[i], im[j] = im[j], im[i]
        k = N // 2
        while k <= j:
            j -= k
            k //= 2
        j += k

    # Каскады (w_im с плюсом)
    for L in range(1, m + 1):
        le = 2 ** L
        le1 = le // 2
        u_re, u_im = 1.0, 0.0
        arg = math.pi / le1
        w_re = math.cos(arg)
        w_im = math.sin(arg)

        for j in range(le1):
            for i in range(j, N, le):
                ip = i + le1
                t_re = re[ip] * u_re - im[ip] * u_im
                t_im = re[ip] * u_im + im[ip] * u_re
                re[ip] = re[i] - t_re
                im[ip] = im[i] - t_im
                re[i] = re[i] + t_re
                im[i] = im[i] + t_im
            temp_u_re = u_re * w_re - u_im * w_im
            u_im = u_re * w_im + u_im * w_re
            u_re = temp_u_re
    return [x / N for x in re]


# --- 2. ГЛАВНАЯ ЛОГИКА И АНАЛИЗ ---

def run_analysis(file_path, channel):
    fs = 2000  # Частота дискретизации 2 кГц
    N_orig = 500
    N_fft = 512

    # Чтение данных
    raw_signal = []
    try:
        with open(file_path, 'rb') as f:
            for _ in range(N_orig):
                chunk = f.read(4)
                if not chunk: break
                raw_signal.append(struct.unpack('f', chunk)[0])
    except FileNotFoundError:
        print(f"Ошибка: Файл {file_path} не найден!")
        return

    # Дополнение нулями
    padded = raw_signal + [0.0] * (N_fft - N_orig)

    # 1. СПЕКТРАЛЬНЫЙ АНАЛИЗ (Прямое БПФ)
    re_spec, im_spec = fft_manual(list(padded))

    # Вычисляем амплитудный спектр: A = sqrt(Re^2 + Im^2)
    # Нормируем на N/2 для корректных вольт
    amp_spectrum = [(math.sqrt(re_spec[i] ** 2 + im_spec[i] ** 2) * 2 / N_fft) for i in range(N_fft // 2)]
    freqs = [(i * fs / N_fft) for i in range(N_fft // 2)]

    # 2. ФИЛЬТРАЦИЯ (Выделение канала)
    f_center = channel * 40  # 1 канал=40Гц, 2=80Гц и т.д.
    df = fs / N_fft

    filt_re = [0.0] * N_fft
    filt_im = [0.0] * N_fft

    for i in range(N_fft // 2):
        freq = i * df
        # Полоса пропускания +/- 20 Гц от центральной
        if (f_center - 20) <= freq <= (f_center + 20):
            filt_re[i] = re_spec[i]
            filt_im[i] = im_spec[i]
            # Зеркальная частота для симметрии
            filt_re[N_fft - i] = re_spec[N_fft - i]
            filt_im[N_fft - i] = im_spec[N_fft - i]

    # Спектр после фильтрации (для визуализации)
    amp_filt_spec = [(math.sqrt(filt_re[i] ** 2 + filt_im[i] ** 2) * 2 / N_fft) for i in range(N_fft // 2)]

    # 3. ВОССТАНОВЛЕНИЕ (Обратное БПФ)
    recovered_signal = ifft_manual(filt_re, filt_im)[:N_orig]

    # 4. ДЕКОДИРОВАНИЕ 5-БИТНОГО КОДА
    bit_code = ""
    for i in range(5):
        segment = recovered_signal[i * 100: (i + 1) * 100]
        bit_code += "1" if max([abs(x) for x in segment]) > 0.5 else "0"

    print(f"--- Результаты анализа ---")
    print(f"Выбранный канал: {channel} (Центр: {f_center} Гц)")
    print(f"Расшифрованный код: {bit_code}")

    # --- 5. ГРАФИКИ ---
    plt.rcParams['toolbar'] = 'None'
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))

    # А) Исходный сигнал
    axs[0].plot(raw_signal, color='gray', alpha=0.7)
    axs[0].set_title("1. Исходный многоканальный сигнал (Временная область)")
    axs[0].grid(True)

    # Б) Спектральный анализ (До фильтрации)
    axs[1].stem(freqs, amp_spectrum, markerfmt=" ", basefmt="b-")
    axs[1].set_title("2. Спектральный анализ: все каналы (Частотная область)")
    axs[1].set_xlim(0, 300)  # Ограничим до 300 Гц для наглядности
    axs[1].set_ylabel("Амплитуда (В)")
    axs[1].grid(True)

    # В) Спектр после фильтрации
    axs[2].stem(freqs, amp_filt_spec, markerfmt=" ", basefmt="r-")
    axs[2].set_title(f"3. Выделенный спектр канала №{channel} ({f_center} Гц)")
    axs[2].set_xlim(0, 300)
    axs[2].grid(True)

    # Г) Результат и код
    axs[3].plot(recovered_signal, color='red')
    axs[3].axhline(y=0.5, color='green', linestyle='--')
    axs[3].axhline(y=-0.5, color='green', linestyle='--')
    axs[3].set_title(f"4. Восстановленный сигнал канала (Код: {bit_code})")
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()


# Запуск программы
run_analysis("filekr5.dat", channel=4)