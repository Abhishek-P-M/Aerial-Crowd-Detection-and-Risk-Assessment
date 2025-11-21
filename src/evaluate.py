
import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="configs/visdrone.yaml")
    p.add_argument("--model", required=True, help="Path to trained model weights (.pt)")
    args = p.parse_args()

    model = YOLO(args.model)
    metrics = model.val(data=args.data)
    # Print key metrics
    print("✅ Evaluation complete.")
    try:
        # Ultralytics provides a Metrics object; we display common ones.
        print("mAP50: ", getattr(metrics.box, "map50", None))
        print("mAP50-95: ", getattr(metrics.box, "map", None))
        print("Precision: ", getattr(metrics.box, "mp", None))
        print("Recall: ", getattr(metrics.box, "mr", None))
    except Exception as e:
        print("Metrics object:", metrics)
        print("Note:", e)

if __name__ == "__main__":
    main()
