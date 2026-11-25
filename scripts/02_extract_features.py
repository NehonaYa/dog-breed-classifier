"""2-қадам: алдын ала үйретілген модельдерден белгілерді (features) бір рет шығарып, дискке сақтау.

Неге бұлай? Мақалада негізгі модель мұздатылған (base_model.trainable = False). Мұздатылған
модельдің шығысы әр эпохада өзгермейді, сондықтан оны бір рет есептеп сақтасақ, жоғарғы
классификаторды (head) оқыту CPU-да секундтарға қысқарады. Нәтиже математикалық тұрғыдан
мақаладағы тәсілмен бірдей (GPU қажет емес).

Кеңейту (augmentation): мақаладағыдай көлденең айналдыру (horizontal flip). train үшін айналдырылған
суреттердің белгілерін де сақтаймыз (--flip train). val/test үшін айналдыру TTA-да қолданылады.

Мысалдар:
    .venv\\Scripts\\python -I scripts/02_extract_features.py --backbones mobilenet_v2 efficientnet_b0
    .venv\\Scripts\\python -I scripts/02_extract_features.py --backbones inception_resnet_v2 --flip train val test
    .venv\\Scripts\\python -I scripts/02_extract_features.py --backbones resnet50 --limit 256   # жылдамдық тесті
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from src.backbones import BACKBONES, build_extractor  # noqa: E402
from src.config import FEATURES_DIR  # noqa: E402
from src.data_utils import load_json, load_splits, save_json  # noqa: E402
from src.imagenet import DOG_LIKE_RANGE, imagenet_columns  # noqa: E402


def make_dataset(paths: list[str], size: int, flip: bool, batch: int) -> tf.data.Dataset:
    def load(path):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (size, size), antialias=True)  # float32, [0, 255]
        if flip:
            img = tf.image.flip_left_right(img)
        return img

    return (tf.data.Dataset.from_tensor_slices(paths)
            .map(load, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch)
            .prefetch(tf.data.AUTOTUNE))


def backbone_params(model, with_top: bool) -> int:
    """ImageNet басының (1000 класты Dense) параметрлерін есептемей, тек "дене" параметрлері."""
    total = model.count_params()
    if with_top:
        total -= model.get_layer("predictions").count_params()
    return int(total)


def measure_latency(model, cfg, runs: int = 15) -> float:
    """Бір суретті (batch=1) өңдеу уақыты, мс (медиана) — сайттағы жағдайға жақын."""
    x = cfg["preprocess"](np.random.uniform(0, 255, (1, cfg["size"], cfg["size"], 3)).astype(np.float32))
    model.predict_on_batch(x)  # алғашқы шақыру (graph құру) есепке кірмейді
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict_on_batch(x)
        times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(times)), 1)


def openvino_runner(model, name: str, device: str):
    """Қосымша жеделдету: Keras моделін OpenVINO-ға айналдырып, Intel GPU/CPU-да іске қосу.

    FP32 дәлдікпен TensorFlow-пен бірдей нәтиже береді (тексерілген: белгілердің салыстырмалы
    айырмасы ~2e-6, болжамдар 100% сәйкес). Сайт бәрібір TensorFlow қолданады.
    """
    import openvino as ov

    sm_dir = FEATURES_DIR.parent / "openvino" / name
    if not sm_dir.exists():
        model.export(str(sm_dir), format="tf_saved_model", verbose=False)
    compiled = ov.Core().compile_model(ov.convert_model(str(sm_dir)), device,
                                       {"INFERENCE_PRECISION_HINT": "f32"})

    def run(x):
        res = compiled(x)
        outs = [res[o] for o in compiled.outputs]
        if len(outs) == 1:
            return outs[0]
        probs = next(o for o in outs if o.shape[-1] == 1000)
        feats = next(o for o in outs if o.shape[-1] != 1000)
        return feats, probs

    return run


def extract(name: str, splits: list[str], flip_splits: list[str], batch: int, limit: int | None,
            device: str = "tf") -> None:
    cfg = BACKBONES[name]
    df = load_splits()
    model = build_extractor(name)
    infer = model.predict_on_batch if device == "tf" else openvino_runner(model, name, device)
    out_dir = FEATURES_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = None
    if cfg["with_top"]:
        synsets = df.drop_duplicates("label").sort_values("label")["synset"].tolist()
        cols = imagenet_columns(synsets)

    timing_path = out_dir / "timing.json"
    timing = load_json(timing_path) if timing_path.exists() else {}
    timing["backbone_params"] = backbone_params(model, cfg["with_top"])
    timing["engine"] = "tensorflow-cpu" if device == "tf" else f"openvino-{device.lower()}-fp32"
    if limit:  # жылдамдық тесті: бір суретті өңдеу уақыты (TensorFlow, CPU)
        print(f"[{name}] параметрлер: {timing['backbone_params']:,}  бір суретке: {measure_latency(model, cfg)} мс")
    else:
        save_json(timing, timing_path)
    for split in splits:
        sub = df[df["split"] == split]
        paths = sub["abspath"].tolist()[:limit] if limit else sub["abspath"].tolist()
        if not limit:
            np.save(out_dir / f"{split}_y.npy", sub["label"].to_numpy(np.int16))
        for flip in ([False, True] if split in flip_splits else [False]):
            tag = f"{split}{'_flip' if flip else ''}"
            if not limit and (out_dir / f"{tag}.npy").exists():
                print(f"[{name}] {tag}: бұрын сақталған, өткізіп жіберемін")
                continue
            feats, probs120, dogness = [], [], []
            t0 = time.perf_counter()
            done = 0
            for xb in make_dataset(paths, cfg["size"], flip, batch):
                xb = cfg["preprocess"](xb.numpy())
                res = infer(xb)
                if cfg["with_top"]:
                    f, p = res
                    probs120.append(p[:, cols].astype(np.float32))
                    dogness.append(p[:, DOG_LIKE_RANGE.start:DOG_LIKE_RANGE.stop].sum(1).astype(np.float32))
                else:
                    f = res
                feats.append(np.asarray(f, dtype=np.float16))
                done += len(xb)
                rate = done / (time.perf_counter() - t0)
                print(f"\r[{name}] {tag}: {done}/{len(paths)}  {rate:5.1f} сурет/с", end="", flush=True)
            elapsed = time.perf_counter() - t0
            print(f"  ({elapsed / 60:.1f} мин)")
            if limit:
                continue
            np.save(out_dir / f"{tag}.npy", np.concatenate(feats))
            if cfg["with_top"]:
                np.save(out_dir / f"{tag}_imagenet120.npy", np.concatenate(probs120))
                np.save(out_dir / f"{tag}_dogness.npy", np.concatenate(dogness))
            timing[tag] = {"images": len(paths), "seconds": round(elapsed, 1),
                           "images_per_sec": round(len(paths) / elapsed, 2)}
            save_json(timing, timing_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES), choices=list(BACKBONES))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--flip", nargs="*", default=["train"], help="айналдырылған нұсқасы да сақталатын бөліктер")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="тек жылдамдықты өлшеу (нәтиже сақталмайды)")
    ap.add_argument("--device", default="tf", choices=["tf", "GPU", "CPU"],
                    help="tf — TensorFlow (CPU); GPU/CPU — OpenVINO арқылы (міндетті емес, жылдамдату үшін)")
    args = ap.parse_args()
    for name in args.backbones:
        extract(name, args.splits, args.flip, args.batch, args.limit, args.device)


if __name__ == "__main__":
    main()
