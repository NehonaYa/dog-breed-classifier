"""ImageNet-1k класс индекстері: Stanford Dogs тұқымдарын ImageNet кластарымен сәйкестендіру.

Stanford Dogs-тағы әр тұқым папкасы WordNet synset-ке сәйкес келеді (мысалы n02085620 = Chihuahua).
ImageNet-1k кластары да synset арқылы белгіленген, сондықтан сәйкестендіру нақты.
"""
import json

import keras

CLASS_INDEX_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
DOG_LIKE_RANGE = range(151, 276)  # 151–268: үй иттері, 269–275: қасқыр, қорқау, динго, қызыл қасқыр т.б.


def synset_to_imagenet_index() -> dict[str, int]:
    path = keras.utils.get_file("imagenet_class_index.json", CLASS_INDEX_URL, cache_subdir="models",
                                file_hash="c2c37ea517e94d9795004a39431a14cb")
    with open(path, encoding="utf-8") as f:
        idx = json.load(f)
    return {v[0]: int(k) for k, v in idx.items()}


def imagenet_columns(synsets_in_label_order: list[str]) -> list[int]:
    """Біздің 120 класқа сәйкес ImageNet бағандарының индекстері (label ретімен)."""
    mapping = synset_to_imagenet_index()
    missing = [s for s in synsets_in_label_order if s not in mapping]
    if missing:
        raise ValueError(f"ImageNet-1k ішінде жоқ synset-тер: {missing}")
    return [mapping[s] for s in synsets_in_label_order]
