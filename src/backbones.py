"""Алдын ала үйретілген (ImageNet) негізгі модельдер тізімі — transfer learning-тің "денесі".

Әр модель үшін: кіріс өлшемі, дұрыс алдын ала өңдеу (preprocess_input) функциясы.
with_top=True болса, ImageNet-тің 1000 класты басын да сақтаймыз: ол бізге
  (а) оқытусыз "zero-shot" салыстыру жасауға,
  (ә) сайтта "суретте ит бар ма?" тексерісін жасауға керек.
"""
import keras
from keras import applications as A


def _scale_0_1(x):
    """Мақаладағыдай: пиксельдерді тек 255-ке бөлу ([0, 1] аралығы)."""
    return x / 255.0


BACKBONES = {
    "mobilenet_v2": {
        "title": "MobileNetV2", "builder": A.MobileNetV2, "size": 224,
        "preprocess": A.mobilenet_v2.preprocess_input, "with_top": True,
    },
    "efficientnet_b0": {
        "title": "EfficientNetB0", "builder": A.EfficientNetB0, "size": 224,
        "preprocess": A.efficientnet.preprocess_input, "with_top": True,
    },
    "resnet50": {
        "title": "ResNet50", "builder": A.ResNet50, "size": 224,
        "preprocess": A.resnet50.preprocess_input, "with_top": True,
    },
    "inception_resnet_v2": {
        "title": "InceptionResNetV2", "builder": A.InceptionResNetV2, "size": 299,
        "preprocess": A.inception_resnet_v2.preprocess_input, "with_top": True,
    },
    # Қосымша (негізгі эксперименттерде іске қосылмаған, CPU-да өте баяу):
    # мақаланы пиксель деңгейінде дәл қайталау — 331x331 кіріс және /255 нормализация
    "irv2_article": {
        "title": "InceptionResNetV2 (мақала: 331px, /255)", "builder": A.InceptionResNetV2, "size": 331,
        "preprocess": _scale_0_1, "with_top": False,
    },
}


def build_extractor(name: str) -> keras.Model:
    """Белгілер (features) шығаратын модель.

    with_top=True: шығысы [GAP белгілері, ImageNet 1000 класс ықтималдығы]
    with_top=False: шығысы тек GAP белгілері.
    """
    cfg = BACKBONES[name]
    if cfg["with_top"]:
        base = cfg["builder"](include_top=True, weights="imagenet")
        feats = base.get_layer("predictions").input  # соңғы Dense алдындағы вектор (GAP шығысы)
        return keras.Model(base.input, [feats, base.output], name=f"{name}_extractor")
    size = cfg["size"]
    return cfg["builder"](include_top=False, weights="imagenet", input_shape=(size, size, 3), pooling="avg")
