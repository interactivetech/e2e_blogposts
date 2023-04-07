from pathlib import Path
from PIL import Image
import os
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/Users/mendeza/data/xview/train_images/', help='path to output dir')
    parser.add_argument('--out_dir', type=str, default='/Users/mendeza/data/xview/train_images_rgb/', help='path to output dir')
    # OUT_DIR = '/Users/mendeza/data/xview/train_images_rgb/'
    args = parser.parse_args()
    OUT_DIR = args.out_dir
    assert args.input_dir is not None
    assert OUT_DIR is not None

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        print(f"Created {OUT_DIR} ...")
    bad_names = list(Path(args.input_dir).glob("._*.tif"))
    if bad_names is not None:
        print("renaming bad named files...")
        print(bad_names)
    print([i.rename(Path(i.stem[2:]+i.suffix)) for i in bad_names])
    fnames = list(Path(args.input_dir).glob("*.tif"))
    for f in tqdm(fnames):
        # print(f.name)
        img = Image.open(f).convert('RGB')
        p = Path(f.name)
        # print(p.with_suffix('.png'))
        img.save(os.path.join(OUT_DIR,p.with_suffix('.png')) )
        # break