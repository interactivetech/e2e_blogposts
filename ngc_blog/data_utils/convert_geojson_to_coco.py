import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from sahi.utils.file import load_json, save_json
from tqdm import tqdm
import argparse

# fix the seed
random.seed(13)


def xview_to_coco(
    train_images_dir=None,
    train_images_dir_rgb=None,
    train_geojson_path=None,
    output_dir=None,
    train_split_rate=0.75,
    xview_class_labels=None,
    category_id_remapping=None,
):
    """
    Converts visdrone-det annotations into coco annotation.
    Args:
        train_images_dir: str
            'train_images' folder directory
        train_images_dir_rgb: str
            'train_images_rgb' folder directory
        train_geojson_path: str
            'xView_train.geojson' file path
        output_dir: str
            Output folder directory
        train_split_rate: bool
            Train split ratio
        category_id_remapping: dict
            Used for selecting desired category ids and mapping them.
            If not provided, xView mapping will be used.
            format: str(id) to str(id)
    """

    # init vars
    category_id_to_name = {}
    with open( xview_class_labels, encoding="utf8") as f:
        lines = f.readlines()
    for line in lines:
        category_id = line.split(":")[0]
        category_name = line.split(":")[1].replace("\n", "")
        category_id_to_name[category_id] = category_name

    # if category_id_remapping is None:
    category_id_remapping = load_json(category_id_remapping)
    # category_id_remapping

    # init coco object
    coco = Coco()
    # append categories
    for category_id, category_name in category_id_to_name.items():
        if category_id in category_id_remapping.keys():
            remapped_category_id = category_id_remapping[category_id]
            coco.add_category(
                CocoCategory(id=int(remapped_category_id), name=category_name)
            )

    # parse xview data
    coords, chips, classes, image_name_to_annotation_ind = get_labels(
        train_geojson_path, train_img_dir=train_images_dir
    )
    image_name_list = get_ordered_image_name_list(image_name_to_annotation_ind)

    # convert xView data to COCO format
    for image_name in tqdm(image_name_list, "Converting xView data into COCO format"):
        # create coco image object
        width, height = Image.open(Path(train_images_dir_rgb) / Path(image_name)).size
        coco_image = CocoImage(file_name=image_name, height=height, width=width)

        annotation_ind_list = image_name_to_annotation_ind[image_name]

        # iterate over image annotations
        for annotation_ind in annotation_ind_list:
            bbox = coords[annotation_ind].tolist()
            # print("bbox: ",bbox)
            category_id = str(int(classes[annotation_ind].item()))
            coco_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            if category_id in category_id_remapping.keys():
                category_name = category_id_to_name[category_id]
                remapped_category_id = category_id_remapping[category_id]
            else:
                continue
            # create coco annotation and append it to coco image
            coco_annotation = CocoAnnotation(
                bbox=coco_bbox,
                category_id=int(remapped_category_id),
                category_name=category_name,
            )
            if coco_annotation.area > 0:
                coco_image.add_annotation(coco_annotation)
        coco.add_image(coco_image)

    result = coco.split_coco_as_train_val(train_split_rate=train_split_rate)

    train_json_path = Path(output_dir) / "train.json"
    val_json_path = Path(output_dir) / "val.json"
    save_json(data=result["train_coco"].json, save_path=train_json_path)
    save_json(data=result["val_coco"].json, save_path=val_json_path)


def get_ordered_image_name_list(image_name_to_annotation_ind: Dict):
    image_name_list: List[str] = list(image_name_to_annotation_ind.keys())

    def get_image_ind(image_name: str):
        return int(image_name.split(".")[0])

    image_name_list.sort(key=get_image_ind)

    return image_name_list


def get_labels(fname,train_img_dir):
    """
    Gets label data from a geojson label file
    Args:
        fname: file path to an xView geojson label file
    Output:
        Returns three arrays: coords, chips, and classes corresponding to the
            coordinates, file-names, and classes for each ground truth.
    Modified from https://github.com/DIUx-xView.
    """
    data = load_json(fname)

    coords = np.zeros((len(data["features"]), 4))
    chips = np.zeros((len(data["features"])), dtype="object")
    classes = np.zeros((len(data["features"])))
    image_name_to_annotation_ind = defaultdict(list)
    tif_names = [i.name for i in list(Path(train_img_dir).glob('*.tif'))]
    print("5.tif: ", '5.tif' in tif_names)
    for i in tqdm(range(len(data["features"])), "Parsing xView data"):
        # print(i,data["features"][i])
        if data["features"][i]["properties"]["bounds_imcoords"] != []:
            b_id = data["features"][i]["properties"]["image_id"]
            # https://github.com/DIUx-xView/xView1_baseline/issues/3
            # print(b_id,b_id in tif_names)
            if b_id in tif_names:
                # print(b_id,data["features"][i]["properties"][
                #             "bounds_imcoords"
                #         ].split(","))
                val = np.array(
                    [
                        int(num)
                        for num in data["features"][i]["properties"][
                            "bounds_imcoords"
                        ].split(",")
                    ]
                )
                # chips[i] = b_id # original code
                p = Path(b_id)
                # print(chips[i],p,Path(b_id))
                # print(p.with_suffix('.png'))
                chips[i] = str(p.with_suffix('.png'))# code to change .tif ending to .png
                classes[i] = data["features"][i]["properties"]["type_id"]

                image_name_to_annotation_ind[chips[i]].append(i)

                if val.shape[0] != 4:
                    print("Issues at %d!" % i)
                else:
                    coords[i] = val
        else:
            chips[i] = "None"
    return coords, chips, classes, image_name_to_annotation_ind


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_images_dir', type=str, default='/Users/mendeza/data/xview/train_images/', help='path to rgb train images')
    #train_images_dir_rgb
    parser.add_argument('--train_images_dir_rgb', type=str, default='/Users/mendeza/data/xview/train_images_rgb/', help='path to rgb train images')
    parser.add_argument('--train_geojson_path', type=str, default='/Users/mendeza/data/xview/xView_train.geojson', help='path to .geojson file')
    parser.add_argument('--output_dir', type=str, default='/Users/mendeza/data/xview/', help='path to output dir')
    parser.add_argument('--train_split_rate', type=float, default=0.75, help='Train / Validation ratio')
    parser.add_argument('--category_id_remapping', type=str, default='/Users/mendeza/Documents/projects/xview-torchvision-coco/utils/category_id_mapping.json', help='json that remaps class labels to indices that are suitable for ML training')
    parser.add_argument('--xview_class_labels', type=str, default='/Users/mendeza/Documents/projects/xview-torchvision-coco/utils/xview_class_labels.txt', help='txt file that maps id to class label')
    # OUT_DIR = '/Users/mendeza/data/xview/train_images_rgb/'
    args = parser.parse_args()
    print(args)
    xview_to_coco(
    train_images_dir=args.train_images_dir,
    train_images_dir_rgb=args.train_images_dir_rgb,
    train_geojson_path=args.train_geojson_path,
    output_dir=args.output_dir,
    train_split_rate=args.train_split_rate,
    category_id_remapping =args.category_id_remapping,
    xview_class_labels=args.xview_class_labels)