import matplotlib.pyplot as plt
import numpy as np


def is_float(number):
    try:
        num = float(number)
        return True
    except ValueError:
        return False


def training_plot(log_path: str, loss_offset=20):
    with open(log_path, 'r') as f:
        lines = f.readlines()

    x = []
    y = []
    checkpoint = []
    AP = [[] for _ in range(6)]
    for line in lines:
        line = line.split()

        if len(line) == 8 and line[5] == 'checkpoint':
            checkpoint.append(int(line[-1][-11:-4]))

        if len(line) == 13 and line[0] == '|' and is_float(line[1]):
            AP[0].append(float(line[1]))
            AP[1].append(float(line[3]))
            AP[2].append(float(line[5]))
            AP[3].append(float(line[7]))
            AP[4].append(float(line[9]))
            AP[5].append(float(line[11]))

        if len(line) > 9 and line[6] == 'iter:':
            x.append(int(line[7]))
            y.append(float(line[9]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(x[loss_offset:], y[loss_offset:])
    ax1.set_yscale('log')
    ax1.title.set_text("Training Loss")
    ax1.set_xlabel('iter')
    ax1.set_ylabel('loss')

    ax2.plot(checkpoint, AP[0], label='AP')
    ax2.plot(checkpoint, AP[1], label='AP50')
    ax2.plot(checkpoint, AP[2], label='AP75')
    ax2.plot(checkpoint, AP[3], label='APs')
    ax2.plot(checkpoint, AP[4], label='APm')
    ax2.plot(checkpoint, AP[5], label='APl')
    ax2.legend()
    ax2.title.set_text("Validation AP")
    ax2.set_xlabel('iter')
    ax2.set_ylabel('AP')

    fig.suptitle(log_path[2:-4], fontsize="x-large")
    return np.array(x), np.array(y), np.array(checkpoint), np.array(AP)


def join_stats(x, y, checkpoint, AP, x2, y2, checkpoint2, AP2):
    x2 = x2 + x[-1]
    x = np.concatenate((x, x2), axis=0)
    y = np.concatenate((y, y2), axis=0)
    checkpoint2 = checkpoint2 + checkpoint[-1]
    checkpoint = np.concatenate((checkpoint, checkpoint2), axis=0)
    AP = np.concatenate((AP, AP2), axis=1)
    return x, y, checkpoint, AP


def plot_logs(logs: list):
    x, y, checkpoint, AP = training_plot(logs[0])
    for log in logs[1:]:
        x2, y2, checkpoint2, AP2 = training_plot(log, loss_offset=0)
        x, y, checkpoint, AP = join_stats(x, y, checkpoint, AP, x2, y2, checkpoint2, AP2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(x[20:], y[20:])
    ax1.set_yscale('log')
    ax1.title.set_text("Training Loss")
    ax1.set_xlabel('iter')
    ax1.set_ylabel('loss')
    # ax1.vlines([41999], 0, 0.5, linestyles='dashed', colors='red', linewidth=0.63)
    ax1.text(13000, 0.35, "Warm-up:\nlr=0.04")
    ax1.text(65000, 0.27, "Main:\nlr=0.1")
    ax1.text(100000, 0.25, "Main:\nlr=0.2")
    ax1.text(123000, 0.22, "Main:\nlr=0.4")
    ax1.text(150000, 0.2, "Main:\nlr=0.4\nbalance")
    ax1.text(153000, 0.115, "Fine Tune")


    ax2.plot(checkpoint, AP[0], label='AP')
    ax2.plot(checkpoint, AP[1], label='AP50')
    ax2.plot(checkpoint, AP[2], label='AP75')
    ax2.plot(checkpoint, AP[3], label='APs')
    ax2.plot(checkpoint, AP[4], label='APm')
    ax2.plot(checkpoint, AP[5], label='APl')
    for i in range(6):
        ax2.text(checkpoint[-1], AP[i][-1], str(round(float(AP[i][-1]), 2)))
    ax2.legend()
    ax2.title.set_text("Validation AP")
    ax2.set_xlabel('iter')
    ax2.set_ylabel('AP')

    fig.suptitle('Complete Supervised Training', fontsize="x-large")
    plt.savefig('./Supervised_Training', dpi=100)


# %% plot one by one
# x, y, checkpoint, AP = training_plot('./log_supervised_lr_0.004_warmup.txt')
# # %%
# x2, y2, checkpoint2, AP2 = training_plot('./log_supervised_lr_0.010.txt', loss_offset=0)
# x, y, checkpoint, AP = join_stats(x, y, checkpoint, AP, x2, y2, checkpoint2, AP2)
# # %%
# x2, y2, checkpoint2, AP2 = training_plot('./log_supervised_lr_0.020.txt', loss_offset=0)
# x, y, checkpoint, AP = join_stats(x, y, checkpoint, AP, x2, y2, checkpoint2, AP2)
# # %%
# x2, y2, checkpoint2, AP2 = training_plot('./log_supervised_lr_0.040.txt', loss_offset=0)
# x, y, checkpoint, AP = join_stats(x, y, checkpoint, AP, x2, y2, checkpoint2, AP2)

# %% plot all together
logs = ['./log_supervised_lr_0.004_warmup.txt',
        './log_supervised_lr_0.010.txt',
        './log_supervised_lr_0.020.txt',
        './log_supervised_lr_0.040.txt',
        './log_supervised_lr_0.040_balance.txt',
        'log_supervised_lr_0.040_superbalance.txt']

plot_logs(logs)
# %%
import json
from collections import defaultdict

with open('./metrics.json', 'r') as f:
    lines = f.readlines()

prev = 0
offset = 0
checkpoint = []
AP = defaultdict(list)
for i, l in enumerate(lines):
    if l[2:9] != "bbox/AP":
        continue
    try:
        data = json.loads(l)
    except:
        print(l, i)
    for k in data:
        AP[k].append(data[k])
    if data['iteration'] + offset < prev:
        offset = prev
    prev = data['iteration'] + offset
    checkpoint.append(prev)
# %%
colors = iter(plt.cm.rainbow(np.linspace(0, 1, 100)))
fig, ax = plt.subplots(figsize=(40, 40))
for i in AP:
    if i[:8] == 'bbox/AP-':
        ax.plot(checkpoint, AP[i], label=i[8:], color=next(colors))
        ax.text(checkpoint[-1], AP[i][-1], i[8:] + ' ' + str(round(float(AP[i][-1]), 3)))

plt.legend()
ax.tick_params(axis='both', which='major', labelsize=40)
ax.tick_params(axis='both', which='minor', labelsize=40)
ax.set_title('100 Class APs', size=60)
ax.set_xlabel('iter', fontsize=60)
ax.set_ylabel('AP', fontsize=60)
plt.savefig('./all_AP', dpi=100)
plt.show()

# %%
