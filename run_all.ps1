# Бүкіл pipeline-ды басынан аяғына дейін қайталау (Windows PowerShell).
# Алдымен: python -m venv .venv ; .\.venv\Scripts\python -m pip install -r requirements.txt
# CPU-да толық орындалуы бірнеше сағат алады (ең ауыры — InceptionResNetV2 белгілерін шығару).
# Intel GPU болса, жылдамдату үшін: pip install openvino, сосын 4-жолдағы командаға --device GPU қосыңыз.

$py = ".\.venv\Scripts\python.exe"

function Step([string]$title, [string[]]$cmd) {
    Write-Host "`n=== $title ===" -ForegroundColor Cyan
    & $py -I @cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "Қате: $title" -ForegroundColor Red; exit $LASTEXITCODE }
}

Step "0. Деректерді жүктеу"          @("scripts/00_download_data.py")
Step "1. Тексеру, тазалау, бөлу"     @("scripts/01_prepare_data.py")
Step "1b. EDA графиктері"            @("scripts/01b_eda.py")
Step "2a. Белгілер: жеңіл модельдер" @("scripts/02_extract_features.py", "--backbones", "mobilenet_v2", "efficientnet_b0", "--flip", "train", "val", "test")
Step "2b. Белгілер: InceptionResNetV2" @("scripts/02_extract_features.py", "--backbones", "inception_resnet_v2", "--flip", "train")
# Қосымша (міндетті емес, CPU-да ~2 сағат): ResNet50 және мақаланың дәл көшірмесі (331px, /255)
# & $py -I scripts/02_extract_features.py --backbones resnet50 irv2_article --flip train
Step "3. Baseline CNN"               @("scripts/03_train_baseline_cnn.py")
Step "4. Классификаторлар (heads)"   @("scripts/04_train_heads.py")
Step "5. Жылдамдықты өлшеу"          @("scripts/05_benchmark_latency.py")
Step "6. Бағалау және графиктер"     @("scripts/06_evaluate.py")

Write-Host "`nДайын. Сайтты іске қосу: $py -m uvicorn web.main:app --port 8000" -ForegroundColor Green
