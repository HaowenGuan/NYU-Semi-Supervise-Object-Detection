import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict


# %%
def is_float(number):
    try:
        num = float(number)
        return True
    except ValueError:
        return False


def training_plot(log_path: str, loss_offset=20, supervise=True):
    with open(log_path, 'r') as f:
        lines = f.readlines()

    x = []
    y = []
    checkpoint = []
    AP = [[] for _ in range(6)]
    teacher = 0
    for line in lines:
        line = line.split()

        if len(line) == 8 and line[5] == 'checkpoint':
            checkpoint.append(int(line[-1][-11:-4]))

        if len(line) == 13 and line[0] == '|' and is_float(line[1]):
            if supervise or teacher:
                AP[0].append(float(line[1]))
                AP[1].append(float(line[3]))
                AP[2].append(float(line[5]))
                AP[3].append(float(line[7]))
                AP[4].append(float(line[9]))
                AP[5].append(float(line[11]))
            teacher ^= 1

        if len(line) > 9 and line[6] == 'iter:':
            x.append(int(line[7]))
            y.append(float(line[9]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(x[loss_offset:], y[loss_offset:])
    ax1.set_yscale('log')
    ax1.title.set_text("Training Loss")
    ax1.set_xlabel('iter')
    ax1.set_ylabel('loss')

    if not supervise:
        checkpoint = [1999] + [3999 + 4000 * i for i in range(8)]
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


def plot_supervised_logs(logs: list):
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

    return x, y, checkpoint, AP


# %% plot all supervise logs together
# logs = ['./log_supervised_lr_0.004_warmup.txt',
#         './log_supervised_lr_0.010.txt',
#         './log_supervised_lr_0.020.txt',
#         './log_supervised_lr_0.040.txt',
#         './log_supervised_lr_0.040_balance.txt',
#         './log_supervised_lr_0.040_superbalance.txt']
#
# x1, y1, checkpoint1, AP1 = plot_supervised_logs(logs)


# %% Read supervised metrics json and plot APs
def plot_supervised_APs(path='./metrics_supervise.json'):
    with open(path, 'r') as f:
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

    # plot supervised AP graph
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, 100)))
    fig, ax = plt.subplots(figsize=(40, 40))
    for i in AP:
        if i[:8] == 'bbox/AP-':
            ax.plot(checkpoint, AP[i], label=i[8:], color=next(colors))
            ax.text(checkpoint[-1], AP[i][-1], i[8:] + ' ' + str(round(float(AP[i][-1]), 3)))

    plt.legend()
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.tick_params(axis='both', which='minor', labelsize=40)
    ax.set_title('Supervise 100 Class APs', size=60)
    ax.set_xlabel('iter', fontsize=60)
    ax.set_ylabel('AP', fontsize=60)
    plt.savefig('./supervise_all_AP', dpi=100)
    plt.show()


# %% plot semi-supervise
def plot_semi_supervised_logs(path='./metrics_semi-supervise.json'):
    with open(path, 'r') as f:
        lines = f.readlines()

    prev = 0
    offset = 0
    iterations = []
    loss = []
    checkpoint = []
    AP_classes = defaultdict(list)
    AP = [[] for _ in range(6)]
    for i, l in enumerate(lines):
        try:
            data = json.loads(l)
        except:
            print(l, i)
            exit(1)
        if len(data) == 236:
            checkpoint.append(data['iteration'])

            AP[0].append(data["bbox/AP"])
            AP[1].append(data["bbox/AP50"])
            AP[2].append(data["bbox/AP75"])
            AP[3].append(data["bbox/APs"])
            AP[4].append(data["bbox/APm"])
            AP[5].append(data["bbox/APl"])
            # AP[6].append(data["bbox_student/AP"])

            for key in data:
                if key[:8] == 'bbox/AP-':
                    AP_classes[key].append(data[key])

        elif len(data) == 24:  # loss
            iterations.append(data['iteration'])
            loss.append(data['total_loss'])
        else:
            print(i, 'line is unknown format. ![Warning]')

    # PLOTTING METRIC GRAPHS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(iterations[20:], loss[20:])
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
    # ax2.plot(checkpoint, AP[6], label='Student-AP')
    ax2.legend()
    ax2.title.set_text("Validation AP")
    ax2.set_xlabel('iter')
    ax2.set_ylabel('AP')

    fig.suptitle(path[2:-5], fontsize="x-large")
    plt.savefig('./Semi-supervised_Training', dpi=100)
    plt.show()

    # PLOTTING AP GRAPHS
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, 100)))

    fig, ax = plt.subplots(figsize=(40, 40))
    for i in AP_classes:
        ax.plot(checkpoint, AP_classes[i], label=i[8:], color=next(colors))
        ax.text(checkpoint[-1], AP_classes[i][-1], i[8:] + ' ' + str(round(float(AP_classes[i][-1]), 3)))

    plt.legend()
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.tick_params(axis='both', which='minor', labelsize=40)
    ax.set_title('Semi-supervise 100 Class APs', size=60)
    ax.set_xlabel('iter', fontsize=60)
    ax.set_ylabel('AP', fontsize=60)
    plt.savefig('./Semi-supervise_all_APs', dpi=100)
    plt.show()

    return iterations, loss, checkpoint, AP


# %%
# x2, y2, checkpoint2, AP2 = plot_semi_supervised_logs(path='./metrics_semi-supervise.json')

# %%
# x, y, checkpoint, AP = join_stats(x1, y1, checkpoint1, AP1, x2, y2, checkpoint2, AP2)


# %%
def plot_supervise_and_semi(x, y, checkpoint, AP, super_checkpoint, super_AP):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.plot(x[20:], y[20:])
    ax1.set_yscale('log')
    ax1.title.set_text("Training Loss")
    ax1.set_xlabel('iter')
    ax1.set_ylabel('loss')
    # ax1.vlines([41999], 0, 0.5, linestyles='dashed', colors='red', linewidth=0.63)
    ax1.text(13000, 0.35, "Warm-up:\nlr=0.04")
    ax1.text(60000, 0.27, "Main:\nlr=0.1")
    ax1.text(95000, 0.24, "Main:\nlr=0.2")
    ax1.text(122000, 0.22, "Main:\nlr=0.4")
    ax1.text(150000, 0.18, "Main:\nlr=0.4\nbalance")
    ax1.text(138000, 0.103, "Supervise:\nFine Tune Loss")
    ax1.text(40000, 0.5, "Left:\nSupervised Training", fontsize=12)
    ax1.text(200000, 0.475, "Right:\nSemi-Supervise\ntraining", fontsize=12)

    ax2.plot(checkpoint, AP[0], label='AP')
    ax2.plot(checkpoint, AP[1], label='AP50')
    ax2.plot(checkpoint, AP[2], label='AP75')
    ax2.plot(checkpoint, AP[3], label='APs')
    ax2.plot(checkpoint, AP[4], label='APm')
    ax2.plot(checkpoint, AP[5], label='APl')
    for i in range(6):
        ax2.text(super_checkpoint[-1] - 10000, super_AP[i][-1] + 1, str(round(float(super_AP[i][-1]), 2)))
    for i in range(6):
        ax2.text(checkpoint[-1], AP[i][-1], str(round(float(AP[i][-1]), 2)))
    ax2.text(super_checkpoint[-1] - 10000, 37.7, "Supervised\nfinal AP:")
    ax2.text(checkpoint[-1], 41.7, "Semi-\nSupervised\nfinal AP:")
    ax2.legend()
    ax2.title.set_text("Validation AP")
    ax2.set_xlabel('iter')
    ax2.set_ylabel('AP')

    fig.suptitle('Supervised and Semi-Supervised Combined Training', fontsize="x-large")
    plt.savefig('./supervise_and_semi', dpi=100)
# %%
# plot_supervise_and_semi(x, y, checkpoint1, AP)