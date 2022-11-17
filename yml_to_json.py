# %% This is a interactive python script, we recommend using a spyder or dataspell iDE.
import yaml
from detectron2.structures import BoxMode
import detectron2.data.detection_utils as utils
import os
import json
from PIL import Image

train_path = "dataset/labeled_data/training/"
val_path = "dataset/labeled_data/validation/"
unlabel_path = "dataset/unlabeled_data/"


# %%
# def get_all_categories(dataset_path):
#     categorys = set()
#     for file in os.listdir(dataset_path + "labels"):
#         with open(dataset_path + "labels/" + file, "r") as stream:
#             label = yaml.safe_load(stream)
#         for category in label['labels']:
#             categorys.add(category)
#     return list(category_list)


# # %%
# category_list = get_all_categories(train_path)

class_dict = {
    "cup or mug": 0, "bird": 1, "hat with a wide brim": 2, "person": 3, "dog": 4, "lizard": 5, "sheep": 6, "wine bottle": 7,
    "bowl": 8, "airplane": 9, "domestic cat": 10, "car": 11, "porcupine": 12, "bear": 13, "tape player": 14, "ray": 15, "laptop": 16,
    "zebra": 17, "computer keyboard": 18, "pitcher": 19, "artichoke": 20, "tv or monitor": 21, "table": 22, "chair": 23,
    "helmet": 24, "traffic light": 25, "red panda": 26, "sunglasses": 27, "lamp": 28, "bicycle": 29, "backpack": 30, "mushroom": 31,
    "fox": 32, "otter": 33, "guitar": 34, "microphone": 35, "strawberry": 36, "stove": 37, "violin": 38, "bookshelf": 39,
    "sofa": 40, "bell pepper": 41, "bagel": 42, "lemon": 43, "orange": 44, "bench": 45, "piano": 46, "flower pot": 47, "butterfly": 48,
    "purse": 49, "pomegranate": 50, "train": 51, "drum": 52, "hippopotamus": 53, "ski": 54, "ladybug": 55, "banana": 56, "monkey": 57,
    "bus": 58, "miniskirt": 59, "camel": 60, "cream": 61, "lobster": 62, "seal": 63, "horse": 64, "cart": 65, "elephant": 66,
    "snake": 67, "fig": 68, "watercraft": 69, "apple": 70, "antelope": 71, "cattle": 72, "whale": 73, "coffee maker": 74, "baby bed": 75,
    "frog": 76, "bathing cap": 77, "crutch": 78, "koala bear": 79, "tie": 80, "dumbbell": 81, "tiger": 82, "dragonfly": 83, "goldfish": 84,
    "cucumber": 85, "turtle": 86, "harp": 87, "jellyfish": 88, "swine": 89, "pretzel": 90, "motorcycle": 91, "beaker": 92, "rabbit": 93,
    "nail": 94, "axe": 95, "salt or pepper shaker": 96, "croquet ball": 97, "skunk": 98, "starfish": 99,
}

# %%
def get_dataset_dicts(dataset_path, class_dict, start=1, end=2, unlabel=False):
    dataset_dicts = {}
    images = []
    annotations = []

    if unlabel:
        print("unlabel", unlabel)
        for file in os.listdir(dataset_path):
            record = {}
            filename = dataset_path + file
            im = utils.read_image(filename, format="BGR")
            width, height = im.shape[1], im.shape[0]
            # width, height = Image.open(filename).size
            record["file_name"] = file
            record["id"] = int(file.split(".")[0]) + 50000
            record["height"] = height
            record["width"] = width
            images.append(record)
        dataset_dicts["images"] = images
        return dataset_dicts

    categories = []
    for c in class_dict.keys():
        category = {
            "id": class_dict[c],
            "name": c
        }
        categories.append(category)

    an_id = 0
    for i in range(start, end):
        file = str(i) + ".yml"
        record = {}
        with open(dataset_path + "labels/" + file, "r") as stream:
            try:
                label = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        filename = dataset_path + "images/" + file.split(".")[0] + ".JPEG"
        im = utils.read_image(filename, format="BGR")
        width, height = im.shape[1], im.shape[0]
        # width, height = Image.open(filename).size
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
            category_id = class_dict[category]
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
train_dict = get_dataset_dicts(train_path, class_dict, start=1, end=30001)
with open("dataset/labeled_data/labeled_train.json", "w") as outfile:
    json.dump(train_dict, outfile)
print('done')
#%%
print(train_dict['images'][544])
# %%
val_dict = get_dataset_dicts(val_path, class_dict, start=30001, end=50001)
with open("dataset/labeled_data/labeled_val.json", "w") as outfile:
    json.dump(val_dict, outfile)
print('done')
# %%
unlabel_dict = get_dataset_dicts(unlabel_path, class_dict, True)
with open("dataset/labeled_data/unlabeled.json", "w") as outfile:
    json.dump(unlabel_dict, outfile)
