"""Сайтта қолданылатын болжау модулі: сурет -> 120 тұқымның ықтималдықтары.

Құрылымы (transfer learning):
    сурет -> өлшемін өзгерту -> алдын ала үйретілген модель (ImageNet, мұздатылған)
          -> белгілер векторы -> біз оқытқан бас (head) -> 120 тұқым ықтималдығы
Қосымша: ImageNet-тің 1000 класты басы "суретте ит бар ма?" деген сұраққа жауап береді.
"""
import io
import time

import joblib
import keras
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from src.backbones import BACKBONES, build_extractor
from src.config import FINAL_DIR
from src.data_utils import load_json
from src.imagenet import DOG_LIKE_RANGE


class BreedPredictor:
    """models/final/ қалтасындағы модельді жүктейді (деректер жиынының өзі қажет емес)."""

    def __init__(self, model_dir=FINAL_DIR):
        info = load_json(model_dir / "model_info.json")
        self.info = info
        self.backbone = info["backbone"]
        self.head_name = info["head"]
        self.tta = bool(info.get("tta"))
        self.cfg = BACKBONES[self.backbone]
        self.names = load_json(model_dir / "labels.json")["names"]
        self.low_conf = float(info.get("low_confidence_threshold", 0.4))
        self.dog_threshold = float(info.get("dogness_threshold", 0.5))
        # Негізгі модель: ImageNet салмақтары бірінші іске қосқанда автоматты жүктеледі (~/.keras/models)
        self.extractor = build_extractor(self.backbone)
        if self.head_name == "logreg":  # бас түрі model_info.json-нан анықталады
            clf = joblib.load(model_dir / "head.joblib")
            self._head = clf.predict_proba
        else:
            head = keras.models.load_model(model_dir / "head.keras")
            self._head = lambda f: head.predict_on_batch(f)
        self.predict(Image.new("RGB", (64, 64)))  # алғашқы шақыру: графты алдын ала құру

    @staticmethod
    def load_image(data: bytes) -> Image.Image:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)  # телефон суреттерінің бұрылысын түзету
        return img.convert("RGB")

    def _prepare(self, img: Image.Image) -> np.ndarray:
        size = self.cfg["size"]
        x = tf.image.resize(np.asarray(img, dtype=np.float32), (size, size), antialias=True).numpy()
        batch = [x, x[:, ::-1, :]] if self.tta else [x]  # TTA: көлденең айналдырылған нұсқа
        return self.cfg["preprocess"](np.stack(batch))

    def predict(self, img: Image.Image, top_k: int = 5) -> dict:
        t0 = time.perf_counter()
        x = self._prepare(img)
        out = self.extractor.predict_on_batch(x)
        feats, imagenet = (out[0], out[1]) if isinstance(out, (list, tuple)) else (out, None)
        probs = np.asarray(self._head(feats)).mean(0)
        dogness = (float(imagenet[:, DOG_LIKE_RANGE.start:DOG_LIKE_RANGE.stop].sum(1).mean())
                   if imagenet is not None else None)
        order = np.argsort(-probs)[:top_k]
        top = [{"breed": self.names[i], "prob": round(float(probs[i]), 4), "label": int(i)} for i in order]
        return {
            "top": top,
            "dogness": None if dogness is None else round(dogness, 4),
            "is_dog": None if dogness is None else dogness >= self.dog_threshold,
            "low_confidence": top[0]["prob"] < self.low_conf,
            "inference_ms": round((time.perf_counter() - t0) * 1000, 1),
            "model": self.info.get("title"),
        }
