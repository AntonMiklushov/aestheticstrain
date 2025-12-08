import numpy as np
import matplotlib.pyplot as plt

# Итоговые накопители
total_counts = np.zeros(10, dtype=np.int64)  # сколько раз встречается каждая оценка 1–10
means = []                                   # средняя оценка каждого изображения

with open("data/AVA/AVA.txt", "r") as f:
    for line in f:
        # Убираем кавычки в начале и конце и разбиваем по пробелам
        parts = line.strip().strip('"').split()

        # counts[i] = сколько человек поставило оценку (i+1)
        cnt = np.array(list(map(int, parts[2:12])))

        # Накопление распределения всех оценок
        total_counts += cnt

        # Средняя оценка изображения
        s = cnt.sum()
        if s > 0:
            mean = (cnt * np.arange(1, 11)).sum() / s
            means.append(mean)

means = np.array(means)

# ===== РАСПРЕДЕЛЕНИЕ СРЕДНИХ ОЦЕНОК =====
plt.figure(figsize=(8, 5))
plt.hist(means, bins=40)
plt.xlabel("Mean score")
plt.ylabel("Number of images")
plt.title("Distribution of mean scores")
plt.tight_layout()
plt.show()

# ===== РАСПРЕДЕЛЕНИЕ ВСЕХ ОЦЕНОК =====
plt.figure(figsize=(8, 5))
plt.bar(np.arange(1, 11), total_counts)
plt.xlabel("Score")
plt.ylabel("Total count")
plt.title("Distribution of all individual scores")
plt.tight_layout()
plt.show()

# Печать статистики
print("Mean score stats:")
print(" count:", len(means))
print(" min  :", means.min())
print(" max  :", means.max())
print(" mean :", means.mean())
print(" std  :", means.std())
