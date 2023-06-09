import os

from sahi.slicing import slice_coco
from sahi.utils.file import Path, save_json
import argparse


def slice(
    image_dir: str,
    dataset_json_path: str,
    slice_size: int = 512,
    overlap_ratio: float = 0.2,
    ignore_negative_samples: bool = False,
    output_dir: str = "runs/slice_coco",
    min_area_ratio: float = 0.1,
):
    """
    Args:
        image_dir (str): directory for coco images
        dataset_json_path (str): file path for the coco dataset json file
        slice_size (int)
        overlap_ratio (float): slice overlap ratio
        ignore_negative_samples (bool): ignore images without annotation
        output_dir (str): output export dir
        min_area_ratio (float): If the cropped annotation area to original
            annotation ratio is smaller than this value, the annotation
            is filtered out. Default 0.1.
    """

    # assure slice_size is list
    slice_size_list = slice_size
    if isinstance(slice_size_list, (int, float)):
        slice_size_list = [slice_size_list]

    # slice coco dataset images and annotations
    print("Slicing step is starting...")
    for slice_size in slice_size_list:
        # in format: train_images_512_01
        output_images_folder_name = (
            Path(dataset_json_path).stem + f"_images_{str(slice_size)}_{str(overlap_ratio).replace('.','')}"
        )
        output_images_dir = str(Path(output_dir) / output_images_folder_name)
        sliced_coco_name = Path(dataset_json_path).name.replace(
            ".json", f"_{str(slice_size)}_{str(overlap_ratio).replace('.','')}"
        )
        coco_dict, coco_path = slice_coco(
            coco_annotation_file_path=dataset_json_path,
            image_dir=image_dir,
            output_coco_annotation_file_name="",
            output_dir=output_images_dir,
            ignore_negative_samples=ignore_negative_samples,
            slice_height=slice_size,
            slice_width=slice_size,
            min_area_ratio=min_area_ratio,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            out_ext=".jpg",
            verbose=False,
        )
        output_coco_annotation_file_path = os.path.join(output_dir, sliced_coco_name + ".json")
        save_json(coco_dict, output_coco_annotation_file_path)
        print(f"Sliced dataset for 'slice_size: {slice_size}' is exported to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='/Users/mendeza/data/xview/train_images_rgb/', help='path to rgb train images')
    parser.add_argument('--train_dataset_json_path', type=str, default='/Users/mendeza/data/xview/train.json', help='path to train json labels in coco format')
    parser.add_argument('--val_dataset_json_path', type=str, default='/Users/mendeza/data/xview/val.json', help='path to train json labels in coco format')
    parser.add_argument('--slice_size', type=int, default=300, help='size of image slice')
    parser.add_argument('--overlap_ratio', type=float, default=0.2, help='The percentage of overlap each tile will have')
    parser.add_argument('--ignore_negative_samples', nargs='?', const=True, default=True, help='Ignore negative tiles (tiles that have no label)')
    parser.add_argument('--min_area_ratio', type=float, default=0.1, help='Threshold that defines if the new bounding box is X% less than min_area_ratio, throw out label')
    parser.add_argument('--output_train_dir', type=str, default='/Users/mendeza/data/xview/train_sliced_no_neg/', help='Directory to export sliced images')
    parser.add_argument('--output_val_dir', type=str, default='/Users/mendeza/data/xview/val_sliced_no_neg/', help='Directory to export sliced images')

    args = parser.parse_args()
    slice(image_dir=args.image_dir,
                dataset_json_path=args.train_dataset_json_path,
                slice_size= args.slice_size,
                overlap_ratio = args.overlap_ratio,
                ignore_negative_samples = args.ignore_negative_samples,
                min_area_ratio = args.min_area_ratio,
                output_dir=args.output_train_dir)
    
    # Helper code if you want to slice validation images
    slice(image_dir=args.image_dir,
                dataset_json_path=args.val_dataset_json_path,
                slice_size= args.slice_size,
                overlap_ratio = args.overlap_ratio,
                ignore_negative_samples = args.ignore_negative_samples,
                min_area_ratio = args.min_area_ratio,
                output_dir=args.output_val_dir)

    # slice_xview(image_dir='/Users/mendeza/data/xview/train_images',
    #             dataset_json_path='/Users/mendeza/data/xview/val.json',
    #             output_dir='/Users/mendeza/data/xview/val_sliced/')