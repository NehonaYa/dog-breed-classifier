"""Сайт: FastAPI сервері (беттер + API).

Іске қосу (жобаның түбірінен):
    python -m uvicorn web.main:app --port 8000
Браузерде ашу: http://127.0.0.1:8000        API құжаттамасы: http://127.0.0.1:8000/docs
"""
import random
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import DATA_QUALITY_JSON, FIG_DIR, METRICS_DIR
from src.data_utils import load_json, load_splits

WEB_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_MB = 10
state: dict = {}
predict_lock = threading.Lock()  # TensorFlow моделін бір уақытта бір сұраныс қолданады


def read_json(path, default):
    try:
        return load_json(path)
    except (FileNotFoundError, ValueError):
        return default


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from src.inference import BreedPredictor  # TensorFlow-ты сервер іске қосылғанда бір рет жүктейміз

    state["predictor"] = BreedPredictor()
    state["final"] = read_json(METRICS_DIR / "final_model.json", {})
    state["summary"] = read_json(METRICS_DIR / "summary.json", {})
    state["comparison"] = read_json(METRICS_DIR / "comparison.json", [])
    state["quality"] = read_json(DATA_QUALITY_JSON, {})
    state["eda"] = read_json(METRICS_DIR / "eda_stats.json", {})
    try:  # "кездейсоқ мысал" батырмасы үшін test суреттері (деректер жүктелген болса)
        df = load_splits()
        test = df[df["split"] == "test"].reset_index(drop=True)
        state["test_df"] = test if Path(test.loc[0, "abspath"]).exists() else None
    except (FileNotFoundError, KeyError):
        state["test_df"] = None
    yield


app = FastAPI(title="Ит тұқымын анықтау — API", version="1.0", lifespan=lifespan,
              description="Суреттен иттің тұқымын анықтайтын модель (transfer learning, 120 тұқым).")
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
app.mount("/figures", StaticFiles(directory=FIG_DIR, check_dir=False), name="figures")
templates = Jinja2Templates(directory=WEB_DIR / "templates")
templates.env.filters["pct"] = lambda v, d=1: "—" if v is None else f"{float(v) * 100:.{d}f}%"
templates.env.globals["asset_v"] = "1.2"  # CSS/JS өзгерсе, нөмірді арттырыңыз (браузер кэшін жаңарту)
templates.env.filters["num"] =lambda v: "—" if v is None else f"{int(v):,}".replace(",", " ")


def page(request: Request, name: str, active: str):
    ctx = {k: state.get(k) for k in ("final", "summary", "comparison", "quality", "eda")}
    ctx.update(active=active, has_test_images=state.get("test_df") is not None)
    return templates.TemplateResponse(request, name, ctx)


@app.get("/", include_in_schema=False)
def index(request: Request):
    return page(request, "index.html", "home")


@app.get("/models", include_in_schema=False)
def models_page(request: Request):
    return page(request, "models.html", "models")


@app.get("/data", include_in_schema=False)
def data_page(request: Request):
    return page(request, "data.html", "data")


@app.get("/about", include_in_schema=False)
def about_page(request: Request):
    return page(request, "about.html", "about")


@app.post("/api/predict", summary="Суреттен тұқымды анықтау")
def predict(file: UploadFile = File(..., description="Ит суреті (JPG, PNG, WEBP)")):
    data = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Файл {MAX_UPLOAD_MB} МБ-тан үлкен")
    predictor = state["predictor"]
    try:
        img = predictor.load_image(data)
    except Exception:
        raise HTTPException(400, "Файлды сурет ретінде оқу мүмкін болмады (JPG, PNG немесе WEBP жіберіңіз)")
    with predict_lock:
        return predictor.predict(img)


@app.get("/api/random-test", summary="Test жиынынан кездейсоқ сурет және оның болжамы")
def random_test():
    test = state.get("test_df")
    if test is None:
        raise HTTPException(404, "Test суреттері бұл компьютерде жоқ (деректер жүктелмеген)")
    idx = random.randrange(len(test))
    row = test.loc[idx]
    predictor = state["predictor"]
    with open(row["abspath"], "rb") as f:
        img = predictor.load_image(f.read())
    with predict_lock:
        result = predictor.predict(img)
    true_breed = predictor.names[int(row["label"])]
    return {**result, "image_url": f"/api/test-image/{idx}", "true_breed": true_breed,
            "correct": result["top"][0]["label"] == int(row["label"])}


@app.get("/api/test-image/{idx}", include_in_schema=False)
def test_image(idx: int):
    test = state.get("test_df")
    if test is None or not 0 <= idx < len(test):
        raise HTTPException(404)
    return FileResponse(test.loc[idx, "abspath"], media_type="image/jpeg")


@app.get("/api/models", summary="Барлық модельдің метрикалары")
def models_api():
    return {"final": state.get("final"), "models": state.get("comparison")}


@app.get("/api/health", summary="Сервер күйі")
def health():
    return {"status": "ok", "model": state.get("final", {}).get("model_id"),
            "test_images": state.get("test_df") is not None}
