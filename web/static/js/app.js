// Басты беттегі демо: суретті серверге жіберу (/api/predict) және нәтижені көрсету.
(() => {
  const cfg = window.APP_CONFIG || { lowConf: 0.4, dogThreshold: 0.5 };
  const MAX_MB = 10;
  const $ = (id) => document.getElementById(id);

  const dropzone = $("dropzone");
  const fileInput = $("file-input");
  const preview = $("preview");
  const dropEmpty = $("drop-empty");
  const randomBtn = $("random-btn");
  const els = {
    empty: $("result-empty"), loading: $("result-loading"), error: $("result-error"), body: $("result-body"),
    top1Name: $("top1-name"), top1Prob: $("top1-prob"), top1Bar: $("top1-bar"), top5: $("top5"),
    truth: $("truth"), warnDog: $("warn-dog"), warnDogText: $("warn-dog-text"),
    warnConf: $("warn-conf"), warnConfText: $("warn-conf-text"), meta: $("meta"),
  };
  let previewUrl = null;
  let busy = false;

  const pct = (p) => `${(p * 100).toFixed(1)}%`;

  function show(state) {
    els.empty.hidden = state !== "empty";
    els.loading.hidden = state !== "loading";
    els.body.hidden = state !== "result";
    els.error.hidden = state !== "error";
    dropzone.setAttribute("aria-busy", state === "loading" ? "true" : "false");
  }

  function setPreview(src) {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = src.startsWith("blob:") ? src : null;
    preview.src = src;
    preview.alt = "Жүктелген сурет";
    preview.hidden = false;
    dropEmpty.hidden = true;
    dropzone.classList.add("has-image");
  }

  function showError(message) {
    els.error.textContent = message;
    show("error");
  }

  function render(data) {
    const top1 = data.top[0];
    els.top1Name.textContent = top1.breed;
    els.top1Prob.textContent = pct(top1.prob);
    requestAnimationFrame(() => { els.top1Bar.style.width = pct(top1.prob); });

    els.top5.innerHTML = "";
    data.top.forEach((t, i) => {
      const li = document.createElement("li");
      li.className = i === 0 ? "top5-item is-top" : "top5-item";
      li.innerHTML = `
        <span class="top5-name"></span>
        <span class="top5-bar" aria-hidden="true"><span class="top5-fill"></span></span>
        <span class="top5-value"></span>`;
      li.querySelector(".top5-name").textContent = t.breed;
      li.querySelector(".top5-value").textContent = pct(t.prob);
      els.top5.appendChild(li);
      requestAnimationFrame(() => { li.querySelector(".top5-fill").style.width = pct(t.prob); });
    });

    if (data.true_breed) {
      els.truth.hidden = false;
      els.truth.className = `truth ${data.correct ? "truth-ok" : "truth-bad"}`;
      els.truth.innerHTML = `<svg class="icon" aria-hidden="true"><use href="#${data.correct ? "i-check" : "i-x"}"/></svg>
        <span>Шын тұқымы: <strong></strong> — ${data.correct ? "модель дұрыс тапты" : "модель қателесті"}</span>`;
      els.truth.querySelector("strong").textContent = data.true_breed;
    } else {
      els.truth.hidden = true;
    }

    const notDog = data.is_dog === false;
    els.warnDog.hidden = !notDog;
    if (notDog) {
      els.warnDogText.textContent = `Суретте ит анықталмады: ImageNet бойынша «ит» ықтималдығы небәрі ${pct(data.dogness)}. ` +
        "Модель тек ит тұқымдарын таниды, сондықтан төмендегі нәтиже сенімсіз.";
    }
    els.warnConf.hidden = !data.low_confidence || notDog;
    if (data.low_confidence) {
      els.warnConfText.textContent = `Модель сенімсіз: ең үлкен ықтималдық ${pct(top1.prob)} (шегі ${pct(cfg.lowConf)}). ` +
        "Бұл аралас тұқым, бір-біріне ұқсас тұқымдар немесе сапасы төмен сурет болуы мүмкін.";
    }
    els.meta.textContent = `Өңдеу уақыты: ${Math.round(data.inference_ms)} мс · Модель: ${data.model || "—"}`;
    show("result");
  }

  async function request(url, options) {
    if (busy) return;
    busy = true;
    show("loading");
    if (randomBtn) randomBtn.disabled = true;
    try {
      const res = await fetch(url, options);
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `Сервер қатесі (${res.status})`);
      if (data.image_url) setPreview(data.image_url);
      render(data);
    } catch (err) {
      showError(err.message || "Белгісіз қате. Серверді тексеріңіз.");
    } finally {
      busy = false;
      if (randomBtn) randomBtn.disabled = false;
    }
  }

  function handleFile(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      showError("Бұл сурет емес. JPG, PNG немесе WEBP файлын таңдаңыз.");
      return;
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      showError(`Файл тым үлкен (${(file.size / 1048576).toFixed(1)} МБ). Шегі — ${MAX_MB} МБ.`);
      return;
    }
    setPreview(URL.createObjectURL(file));
    const form = new FormData();
    form.append("file", file);
    request("/api/predict", { method: "POST", body: form });
  }

  const openPicker = () => fileInput.click();
  dropzone.addEventListener("click", openPicker);
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openPicker(); }
  });
  $("pick-btn").addEventListener("click", openPicker);
  fileInput.addEventListener("change", () => { handleFile(fileInput.files[0]); fileInput.value = ""; });

  ["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("is-dragover");
  }));
  ["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("is-dragover");
  }));
  dropzone.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));
  document.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) handleFile(item.getAsFile());
  });
  if (randomBtn) randomBtn.addEventListener("click", () => request("/api/random-test"));
  // ?example=1 — бет ашылғанда test жиынынан мысалды бірден көрсету (демо үшін)
  if (randomBtn && new URLSearchParams(location.search).has("example")) request("/api/random-test");
})();
