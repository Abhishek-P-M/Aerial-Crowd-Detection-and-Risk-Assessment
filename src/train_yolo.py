import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="configs/visdrone.yaml")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    # 🔥 Early stopping patience: stop if best val metric doesn't improve for N epochs
    p.add_argument("--patience", type=int, default=5)

    # ✅ NEW: save checkpoint every N epochs (-1 = default behaviour)
    p.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save checkpoint every N epochs (-1 = only best/last)"
    )

    # New args for resuming
    p.add_argument(
        "--project",
        default="runs/detect",
        help="Root dir to save training runs"
    )
    p.add_argument(
        "--name",
        default="visdrone_person",
        help="Run name (subfolder under project)"
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint of this run"
    )

    args = p.parse_args()

    run_dir = Path(args.project) / args.name
    ckpt_path = run_dir / "weights" / "last.pt"

    if args.resume:
        # ---- Resume from last checkpoint ----
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"⚠️ Cannot resume: checkpoint not found at {ckpt_path}. "
                f"Run once without --resume to start training."
            )

        print(f"🔁 Resuming training from: {ckpt_path}")
        model = YOLO(str(ckpt_path))

        # When resume=True, Ultralytics reloads previous training settings
        results = model.train(
            resume=True,
            save_period=args.save_period  # (may or may not be used, but harmless)
        )

    else:
        # ---- Fresh training run ----
        print(f"🚀 Starting new training run: project={args.project}, name={args.name}")
        model = YOLO(args.model)

        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            patience=args.patience,
            project=args.project,
            name=args.name,
            save_period=args.save_period,  # ✅ important line
        )

    print("✅ Training complete.")
    # results.best is usually available in recent Ultralytics versions
    if hasattr(results, "best"):
        print("🏆 Best weights:", results.best)
    else:
        print("ℹ️ Training finished (no results.best attribute found).")


if __name__ == "__main__":
    main()
