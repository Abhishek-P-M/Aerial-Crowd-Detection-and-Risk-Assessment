
import argparse, os
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to weights .pt (e.g., runs/detect/train/weights/best.pt)")
    p.add_argument("--source", required=True, help="Folder of images (e.g., data/visdrone-yolo/images/val)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--name", default="predict-visdrone")
    p.add_argument("--save", action="store_true", help="Save visualized predictions")
    args = p.parse_args()

    model = YOLO(args.model)
    results = model.predict(source=args.source, conf=args.conf, save=args.save, name=args.name)
    print("✅ Prediction done. See runs/detect/%s" % args.name)

if __name__ == "__main__":
    main()
