import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


METRIC_FILE_NAMES = {"results.csv", "history.csv", "history.json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YOLO training metrics")
    parser.add_argument(
        "--source",
        nargs="+",
        default=["runs/classify"],
        help="One or more run directories or metrics files to compare (.csv/.json)",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional display labels matching --source order",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to <source>/visualizations/training_curves_compare.png when multiple sources are used",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Only visualize the newest metrics file for each run directory",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving it",
    )
    return parser.parse_args()


def normalize_label(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("(", "")
        .replace(")", "")
    )


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {normalize_label(column): column for column in columns}
    for candidate in candidates:
        key = normalize_label(candidate)
        if key in normalized:
            return normalized[key]

    for column in columns:
        normalized_column = normalize_label(column)
        for candidate in candidates:
            if normalize_label(candidate) in normalized_column:
                return column
    return None


def load_metrics_file(metrics_file: Path) -> pd.DataFrame:
    if metrics_file.suffix.lower() == ".json":
        raw_frame = pd.read_json(metrics_file)
    else:
        raw_frame = pd.read_csv(metrics_file)

    if raw_frame.empty:
        return raw_frame

    frame = raw_frame.copy()
    rename_map = {}
    epoch_column = find_column(frame.columns, ["epoch", "epochs"])
    train_loss_column = find_column(frame.columns, ["train_loss", "loss", "train/loss", "loss/train"])
    train_acc_column = find_column(frame.columns, ["train_acc", "acc", "accuracy", "train/acc", "accuracy/train"])
    val_loss_column = find_column(frame.columns, ["val_loss", "valid_loss", "metrics/val_loss", "validation_loss"])
    val_acc_column = find_column(frame.columns, ["val_acc", "valid_acc", "metrics/val_acc", "metrics/accuracy", "val/acc"])
    lr_column = find_column(frame.columns, ["lr", "learning_rate", "train/lr", "lrs"])

    if epoch_column is not None:
        rename_map[epoch_column] = "epoch"
    if train_loss_column is not None:
        rename_map[train_loss_column] = "train_loss"
    if train_acc_column is not None:
        rename_map[train_acc_column] = "train_acc"
    if val_loss_column is not None:
        rename_map[val_loss_column] = "val_loss"
    if val_acc_column is not None:
        rename_map[val_acc_column] = "val_acc"
    if lr_column is not None:
        rename_map[lr_column] = "lr"

    frame = frame.rename(columns=rename_map)

    if "epoch" not in frame.columns:
        frame["epoch"] = range(1, len(frame) + 1)

    return frame


def collect_metric_files(source: Path, latest_only: bool) -> list[Path]:
    if source.is_file():
        return [source]

    metric_files = [path for path in source.rglob("*") if path.is_file() and path.name in METRIC_FILE_NAMES]
    if not latest_only:
        return sorted(metric_files)

    grouped: dict[Path, Path] = {}
    for metric_file in metric_files:
        run_dir = metric_file.parent
        current = grouped.get(run_dir)
        if current is None or metric_file.stat().st_mtime > current.stat().st_mtime:
            grouped[run_dir] = metric_file

    return sorted(grouped.values())


def build_run_name(source_root: Path, metrics_file: Path) -> str:
    if source_root.is_file():
        return metrics_file.stem

    try:
        relative = metrics_file.parent.relative_to(source_root)
        if str(relative) == ".":
            return metrics_file.parent.name
        return str(relative).replace("\\", "/")
    except ValueError:
        return metrics_file.parent.name


def collect_frames_for_source(source: Path, latest_only: bool) -> list[tuple[str, pd.DataFrame]]:
    metric_files = collect_metric_files(source, latest_only)
    frames: list[tuple[str, pd.DataFrame]] = []

    for metric_file in metric_files:
        try:
            frame = load_metrics_file(metric_file)
        except Exception as exc:  # pragma: no cover - explicit error path for user data
            print(f"Skip {metric_file}: {exc}")
            continue

        if frame.empty:
            continue

        frames.append((build_run_name(source, metric_file), frame))

    return frames


def plot_metrics(frames: list[tuple[str, pd.DataFrame]], output_path: Path, show: bool) -> None:
    if not frames:
        raise FileNotFoundError("No metrics files were found to visualize")

    plt.style.use("seaborn-v0_8-darkgrid")
    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    loss_axis = axes[0, 0]
    acc_axis = axes[0, 1]
    lr_axis = axes[1, 0]
    summary_axis = axes[1, 1]

    summary_axis.axis("off")
    summary_lines = ["Training summary"]

    for run_name, frame in frames:
        epochs = frame["epoch"].tolist()

        if "train_loss" in frame.columns:
            loss_axis.plot(epochs, frame["train_loss"], marker="o", linewidth=1.8, label=f"{run_name} train")
        if "val_loss" in frame.columns:
            loss_axis.plot(epochs, frame["val_loss"], marker="s", linewidth=1.8, label=f"{run_name} val")

        if "train_acc" in frame.columns:
            acc_axis.plot(epochs, frame["train_acc"], marker="o", linewidth=1.8, label=f"{run_name} train")
        if "val_acc" in frame.columns:
            acc_axis.plot(epochs, frame["val_acc"], marker="s", linewidth=1.8, label=f"{run_name} val")

        if "lr" in frame.columns:
            lr_axis.plot(epochs, frame["lr"], marker="o", linewidth=1.8, label=run_name)

        best_val_acc = frame["val_acc"].max() if "val_acc" in frame.columns else None
        final_val_acc = frame["val_acc"].iloc[-1] if "val_acc" in frame.columns else None
        final_train_acc = frame["train_acc"].iloc[-1] if "train_acc" in frame.columns else None
        summary_lines.append(
            f"{run_name}: best val acc={best_val_acc:.4f}" if best_val_acc is not None else f"{run_name}: no val acc column"
        )
        if final_val_acc is not None:
            summary_lines.append(f"  final val acc={final_val_acc:.4f}")
        if final_train_acc is not None:
            summary_lines.append(f"  final train acc={final_train_acc:.4f}")

    loss_axis.set_title("Loss Curves")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    if loss_axis.has_data():
        loss_axis.legend(loc="best")

    acc_axis.set_title("Accuracy Curves")
    acc_axis.set_xlabel("Epoch")
    acc_axis.set_ylabel("Accuracy")
    if acc_axis.has_data():
        acc_axis.legend(loc="best")

    lr_axis.set_title("Learning Rate")
    lr_axis.set_xlabel("Epoch")
    lr_axis.set_ylabel("LR")
    if lr_axis.has_data():
        lr_axis.legend(loc="best")

    summary_axis.text(
        0.0,
        1.0,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    figure.suptitle("YOLO Training Comparison" if len(frames) > 1 else "YOLO Training Visualization", fontsize=16, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(figure)


def main() -> None:
    args = parse_args()
    frames: list[tuple[str, pd.DataFrame]] = []

    sources = [Path(source_text) for source_text in args.source]
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(f"Source path not found: {source}")
        frames.extend(collect_frames_for_source(source, args.latest))

    if not frames:
        raise FileNotFoundError(
            f"No usable metric files found under: {', '.join(str(source) for source in sources)}. Expected one of: {', '.join(sorted(METRIC_FILE_NAMES))}"
        )

    if args.output is not None:
        output_path = Path(args.output)
    elif len(sources) == 1 and sources[0].is_file():
        output_path = sources[0].with_name(f"{sources[0].stem}_training_curves.png")
    elif len(sources) == 1:
        output_path = sources[0] / "visualizations" / "training_curves.png"
    else:
        output_path = Path("runs/classify/visualizations/training_curves_compare.png")

    if args.labels:
        if len(args.labels) != len(sources):
            raise ValueError("--labels count must match --source count")
        labeled_frames: list[tuple[str, pd.DataFrame]] = []
        for source_index, source in enumerate(sources):
            source_frames = collect_frames_for_source(source, args.latest)
            run_label = args.labels[source_index]
            for _, frame in source_frames:
                labeled_frames.append((run_label, frame))
        frames = labeled_frames

    plot_metrics(frames, output_path, args.show)
    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    main()