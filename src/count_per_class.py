
import argparse, os, glob, csv
from collections import Counter, defaultdict
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to weights .pt")
    p.add_argument("--source", required=True, help="Folder of images to analyze")
    p.add_argument("--out", default="outputs/counts.csv")
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model = YOLO(args.model)

    # Collect image paths
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    images = [p for p in glob.glob(os.path.join(args.source, "*")) if p.lower().endswith(exts)]
    rows = []
    class_names = [model.names[k] for k in sorted(model.names.keys())]

    for img in images:
        r = model.predict(img, conf=args.conf, verbose=False)[0]
        cnt = Counter()
        for c in r.boxes.cls.tolist():
            cnt[model.names[int(c)]] += 1
        row = {"image": os.path.basename(img)}
        for n in class_names:
            row[n] = cnt.get(n, 0)
        rows.append(row)

    # Write CSV
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image"] + class_names)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"✅ Wrote counts to {args.out}")

if __name__ == "__main__":
    main()
