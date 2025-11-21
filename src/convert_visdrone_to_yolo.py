
import os, argparse, glob, shutil
import cv2

# VisDrone categories documented for DET:
# 0: ignored regions (skip), 1..11: actual categories (we include 11 as 'others')
ID2NAME = {
    1:"pedestrian", 2:"people", 3:"bicycle", 4:"car", 5:"van",
    6:"truck", 7:"tricycle", 8:"awning-tricycle", 9:"bus", 10:"motor", 11:"others"
}
CLS_MAP = {k:i for i,k in enumerate(ID2NAME.keys())}  # 1->0, 2->1, ..., 11->10

def convert_split(split_root, out_root, split_name):
    img_dir = os.path.join(split_root, f"VisDrone2019-DET-{split_name}", "images")
    ann_dir = os.path.join(split_root, f"VisDrone2019-DET-{split_name}", "annotations")
    if not os.path.isdir(img_dir) or not os.path.isdir(ann_dir):
        raise FileNotFoundError(f"Expecting {img_dir} and {ann_dir}")

    out_img = os.path.join(out_root, "images", split_name)
    out_lbl = os.path.join(out_root, "labels", split_name)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
    for ip in img_paths:
        base = os.path.basename(ip)
        shutil.copy(ip, os.path.join(out_img, base))

        ann_in = os.path.join(ann_dir, base.replace(".jpg", ".txt"))
        ann_out = os.path.join(out_lbl, base.replace(".jpg", ".txt"))
        lines_out = []
        if os.path.exists(ann_in):
            im = cv2.imread(ip)
            if im is None:
                # corrupted image; skip with empty label file
                open(ann_out, "w").close()
                continue
            h, w = im.shape[:2]
            with open(ann_in, "r") as f:
                for raw in f.read().strip().splitlines():
                    if not raw.strip():
                        continue
                    parts = raw.split(",")
                    if len(parts) < 6:
                        continue
                    try:
                        x = float(parts[0]); y = float(parts[1])
                        bw = float(parts[2]); bh = float(parts[3])
                        score = float(parts[4])
                        cat = int(float(parts[5]))
                    except Exception:
                        continue

                    if cat == 0:       # ignored region
                        continue
                    if score == 0:     # GT marked to ignore
                        continue
                    if cat not in CLS_MAP:
                        continue
                    if bw <= 0 or bh <= 0:
                        continue

                    xc = (x + bw/2.0) / w
                    yc = (y + bh/2.0) / h
                    nw = bw / w
                    nh = bh / h
                    # Keep only valid normalized boxes
                    if not (0 <= xc <= 1 and 0 <= yc <= 1):
                        continue
                    if nw <= 0 or nh <= 0:
                        continue

                    lines_out.append(f"{CLS_MAP[cat]} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        with open(ann_out, "w") as g:
            g.write("\n".join(lines_out))

    return out_root

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visdrone-root", required=True, help="Folder containing VisDrone2019-DET-train and -val")
    ap.add_argument("--out", default="data/visdrone-yolo", help="Output root folder for YOLO dataset")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)

    convert_split(args.visdrone_root, out_root, "train")
    convert_split(args.visdrone_root, out_root, "val")

    # Also write a names file (optional)
    names_path = os.path.join(out_root, "names.txt")
    with open(names_path, "w") as f:
        for k in sorted(ID2NAME):
            f.write(ID2NAME[k] + "\n")

    print(f"✅ Converted to YOLO at: {out_root}")
    print(f"Images: {os.path.join(out_root, 'images')}")
    print(f"Labels: {os.path.join(out_root, 'labels')}")
    print(f"Class names: {names_path}")

if __name__ == "__main__":
    main()
