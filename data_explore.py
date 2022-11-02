import yaml
import matplotlib.pyplot as plt

# i = [1, 30000]
i = 1

with open("dataset/labeled_data/training/labels/" + str(i) + ".yml", "r") as stream:
    try:
        label = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

img = plt.imread("dataset/labeled_data/training/images/" + str(i) + ".JPEG")

for index in range(len(label['labels'])):
    x1, y1, x2, y2 = label['bboxes'][index]
    name = label['labels'][index]
    plt.imshow(img[y1:y2, x1:x2])
    plt.title(name)
    plt.show()
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
