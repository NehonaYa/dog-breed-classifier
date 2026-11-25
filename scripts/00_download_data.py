"""0-қадам: Stanford Dogs деректерін ресми сайттан жүктеу.

Дереккөз: http://vision.stanford.edu/aditya86/ImageNetDogs/
  images.tar      ~ 757 MB  (20 580 сурет, 120 тұқым)
  annotation.tar  ~ 21 MB   (әр суреттегі иттің bounding box-ы)
  lists.tar       ~ 0.5 MB  (ресми train/test бөлінісі)

Kaggle-дағы "Dog Breed Identification" жарысының деректері (GeeksforGeeks мақаласында
қолданылған) осы Stanford Dogs жиынтығының бір бөлігі: тұқымдар тізімі бірдей.

Іске қосу:
    .venv\\Scripts\\python -I scripts/00_download_data.py
"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ANNOT_DIR, DATA_RAW, IMAGES_DIR  # noqa: E402

BASE_URL = "https://vision.stanford.edu/aditya86/ImageNetDogs/"
# архив -> ашылғаннан кейін пайда болатын белгі (бар болса, қайта жүктемейміз)
FILES = {"lists.tar": DATA_RAW / "train_list.mat", "annotation.tar": ANNOT_DIR, "images.tar": IMAGES_DIR}


def download(url: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dst.name}: {done / 2**20:6.1f} / {total / 2**20:.1f} MB", end="", flush=True)
    tmp.replace(dst)
    print()


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for name, extracted in FILES.items():
        dst = DATA_RAW / name
        if extracted.exists() or (dst.exists() and dst.stat().st_size > 0):
            print(f"Бар: {dst.name}")
            continue
        print(f"Жүктелуде: {BASE_URL + name}")
        download(BASE_URL + name, dst)
    print("Дайын. Келесі қадам: scripts/01_prepare_data.py")


if __name__ == "__main__":
    main()
