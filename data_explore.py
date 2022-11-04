import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# i = [1, 30000]
i = 1

with open("dataset/labeled_data/training/labels/" + str(i) + ".yml", "r") as stream:
    try:
        label = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


fig, ax = plt.subplots()
img = plt.imread("dataset/labeled_data/training/images/" + str(i) + ".JPEG")
ax.imshow(img)

n = len(label['labels'])
colors = iter(plt.cm.rainbow(np.linspace(0, 1, n)))
for index in range(n):
    x1, y1, x2, y2 = label['bboxes'][index]
    name = label['labels'][index]
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=next(colors), facecolor='none', label=name)
    ax.add_patch(rect)
ax.legend()
#%%
max_labels = 0
for i in range(1, 30001):
    with open("dataset/labeled_data/training/labels/" + str(i) + ".yml", "r") as stream:
        try:
            label = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    if len(label['labels']) == 19:
        print(i)
    max_labels = max(max_labels, len(label['labels']))
#%%
