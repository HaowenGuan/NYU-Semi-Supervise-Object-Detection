# %% This is a interactive python script, we recommend using a spyder or dataspell iDE.
import yaml
from detectron2.structures import BoxMode
import os
import cv2
import json
from PIL import Image

train_path = "dataset/labeled_data/training/"
val_path = "dataset/labeled_data/validation/"
unlabel_path = "dataset/unlabeled_data/"


# %%
def get_all_categories(dataset_path):
    categorys = set()
    for file in os.listdir(dataset_path + "labels"):
        with open(dataset_path + "labels/" + file, "r") as stream:
            label = yaml.safe_load(stream)
        for category in label['labels']:
            categorys.add(category)
    return list(category_list)


# %%
category_list = get_all_categories(train_path)


# %%
def get_dataset_dicts(dataset_path, category_list, start=1, end=2, unlabel=False):
    dataset_dicts = {}
    images = []
    annotations = []

    if unlabel:
        for file in os.listdir(dataset_path):
            record = {}
            filename = dataset_path + file
            height, width = cv2.imread(filename).shape[:2]
            record["file_name"] = file
            record["id"] = int(file.split(".")[0]) + 50000
            record["height"] = height
            record["width"] = width
            images.append(record)
        dataset_dicts["images"] = images
        return dataset_dicts

    categories = []
    category_list_index = dict()
    for c in range(len(category_list)):
        category = {
            "id": c + 1,
            "name": category_list[c]
        }
        categories.append(category)
        category_list_index[category_list[c]] = c + 1

    an_id = 0
    for i in range(start, end):
        if i == 545:
            i += 1
            continue
        file = str(i) + ".yml"
        record = {}
        with open(dataset_path + "labels/" + file, "r") as stream:
            try:
                label = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        filename = dataset_path + "images/" + file.split(".")[0] + ".JPEG"
        height, width = cv2.imread(filename).shape[:2]
        # width, height  = label['image_size'][0], label['image_size'][1]
        record["file_name"] = file.split(".")[0] + ".JPEG"
        record["id"] = i - 1
        record["height"] = height
        record["width"] = width

        for j in range(len(label['bboxes'])):
            box = label['bboxes'][j]
            box[2] -= box[0]
            box[3] -= box[1]
            area = box[2] * box[3]
            category = label['labels'][j]
            category_id = category_list_index[category]
            obj = {
                "id": an_id,
                "image_id": i - 1,
                "category_id": category_id,
                "area": area,
                "bbox": box,
                "iscrowd": 0
            }
            annotations.append(obj)
            an_id += 1
        images.append(record)
    dataset_dicts["categories"] = categories
    dataset_dicts["images"] = images
    dataset_dicts["annotations"] = annotations
    return dataset_dicts


# %%
train_dict = get_dataset_dicts(train_path, category_list, start=1, end=30001)
with open("dataset/labeled_data/labeled_train.json", "w") as outfile:
    json.dump(train_dict, outfile)
print('done')
#%%
print(train_dict['images'][544])
# %%
val_dict = get_dataset_dicts(val_path, category_list, start=30001, end=50001)
with open("dataset/labeled_data/labeled_val.json", "w") as outfile:
    json.dump(val_dict, outfile)
print('done')
# %%
unlabel_dict = get_dataset_dicts(unlabel_path, category_list, True)
with open("dataset/labeled_data/unlabeled.json", "w") as outfile:
    json.dump(unlabel_dict, outfile)
