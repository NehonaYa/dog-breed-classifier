// Басты беттегі демо: суретті серверге жіберу (/api/predict) және нәтижені анимациямен көрсету.
(() => {
  const cfg = window.APP_CONFIG || { lowConf: 0.5, dogThreshold: 0.5 };
  const MAX_MB = 10;
  const RING_C = 2 * Math.PI * 52; // дөңгелектің ұзындығы (r = 52)
  const $ = (id) => document.getElementById(id);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const canAnimate = () => window.gsap && !reduceMotion;

  const dropzone = $("dropzone");
  const fileInput = $("file-input");
  const preview = $("preview");
  const dropEmpty = $("drop-empty");
  const scan = $("scan");
  const randomBtn = $("random-btn");
  const resultCard = document.querySelector(".result-card");
  const els = {
    empty: $("result-empty"), loading: $("result-loading"), error: $("result-error"), body: $("result-body"),
    top1Name: $("top1-name"), top1Prob: $("top1-prob"), ring: $("ring-fg"), top5: $("top5"),
    truth: $("truth"), warnDog: $("warn-dog"), warnDogText: $("warn-dog-text"),
    warnConf: $("warn-conf"), warnConfText: $("warn-conf-text"), meta: $("meta"),
  };
  let previewUrl = null;
  let busy = false;

  const pct = (p) => `${(p * 100).toFixed(1)}%`;
  els.ring.style.strokeDasharray = `${RING_C}`;
  els.ring.style.strokeDashoffset = `${RING_C}`;

  function show(state) {
    els.empty.hidden = state !== "empty";
    els.loading.hidden = state !== "loading";
    els.body.hidden = state !== "result";
    els.error.hidden = state !== "error";
    scan.hidden = state !== "loading" || preview.hidden;
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
    if (canAnimate()) gsap.fromTo(preview, { autoAlpha: 0, scale: 0.97 }, { autoAlpha: 1, scale: 1, duration: 0.4, ease: "power2.out" });
  }

  function showError(message) {
    els.error.textContent = message;
    show("error");
  }

  function ringColor(p) {
    // жоғары сенімділік — көк, төмен — қызғылт сары (мәтінмен бірге, тек түске сүйенбейді)
    return p >= cfg.lowConf ? "var(--series-1)" : "var(--accent)";
  }

  function render(data, truthBreed) {
    const top1 = data.top[0];
    els.top1Name.textContent = top1.breed;
    els.ring.style.stroke = ringColor(top1.prob);

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
      li.querySelector(".top5-fill").dataset.w = String(t.prob * 100);
      els.top5.appendChild(li);
    });

    const truth = data.true_breed || truthBreed;
    if (truth) {
      const correct = data.correct ?? (top1.breed === truth);
      els.truth.hidden = false;
      els.truth.className = `truth ${correct ? "truth-ok" : "truth-bad"}`;
      els.truth.innerHTML = `<svg class="icon" aria-hidden="true"><use href="#${correct ? "i-check" : "i-x"}"/></svg>
        <span>Шын тұқымы: <strong></strong>, ${correct ? "модель дұрыс тапты" : "модель қателесті"}</span>`;
      els.truth.querySelector("strong").textContent = truth;
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
    animateResult(top1.prob);
  }

  function animateResult(p) {
    const fills = [...els.top5.querySelectorAll(".top5-fill")];
    const targetOffset = RING_C * (1 - p);
    if (!canAnimate()) {
      els.ring.style.strokeDashoffset = `${targetOffset}`;
      els.top1Prob.textContent = pct(p);
      fills.forEach((f) => { f.style.width = `${f.dataset.w}%`; });
      return;
    }
    const counter = { v: 0 };
    const tl = gsap.timeline({ defaults: { ease: "power3.out" } });
    tl.fromTo(els.body, { autoAlpha: 0, y: 12 }, { autoAlpha: 1, y: 0, duration: 0.35 })
      .fromTo(els.ring, { strokeDashoffset: RING_C }, { strokeDashoffset: targetOffset, duration: 1.1 }, 0.05)
      .to(counter, { v: p * 100, duration: 1.1, onUpdate: () => { els.top1Prob.textContent = `${counter.v.toFixed(1)}%`; } }, 0.05)
      .fromTo(els.top5.children, { autoAlpha: 0, x: -10 }, { autoAlpha: 1, x: 0, duration: 0.35, stagger: 0.06 }, 0.2)
      .fromTo(fills, { width: "0%" }, { width: (i, el) => `${el.dataset.w}%`, duration: 0.8, stagger: 0.06 }, 0.3);
  }

  async function request(url, options, truthBreed) {
    if (busy) return;
    busy = true;
    show("loading");
    if (randomBtn) randomBtn.disabled = true;
    try {
      const res = await fetch(url, options);
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `Сервер қатесі (${res.status})`);
      if (data.image_url) setPreview(data.image_url);
      render(data, truthBreed);
    } catch (err) {
      showError(err.message || "Белгісіз қате. Серверді тексеріңіз.");
    } finally {
      busy = false;
      if (randomBtn) randomBtn.disabled = false;
    }
  }

  function sendFile(file, truthBreed) {
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
    request("/api/predict", { method: "POST", body: form }, truthBreed);
  }

  function scrollToDemo() {
    const target = window.innerWidth < 900 ? resultCard : document.getElementById("demo");
    target.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  }

  // Мысалдар галереясы: суретті алып, /api/predict-ке жібереміз (шын тұқымы белгілі)
  document.querySelectorAll(".ex-card").forEach((card) => {
    card.addEventListener("click", async () => {
      if (busy) return;
      document.querySelectorAll(".ex-card").forEach((c) => c.classList.toggle("is-active", c === card));
      scrollToDemo();
      try {
        const blob = await (await fetch(card.dataset.src)).blob();
        sendFile(new File([blob], card.dataset.src.split("/").pop(), { type: blob.type || "image/jpeg" }), card.dataset.breed);
      } catch {
        showError("Мысал суретті жүктеу мүмкін болмады.");
      }
    });
  });

  const openPicker = () => fileInput.click();
  dropzone.addEventListener("click", openPicker);
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openPicker(); }
  });
  $("pick-btn").addEventListener("click", openPicker);
  fileInput.addEventListener("change", () => { sendFile(fileInput.files[0]); fileInput.value = ""; });

  ["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("is-dragover");
  }));
  ["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("is-dragover");
  }));
  dropzone.addEventListener("drop", (e) => sendFile(e.dataTransfer.files[0]));
  document.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) sendFile(item.getAsFile());
  });
  if (randomBtn) randomBtn.addEventListener("click", () => request("/api/random-test"));
  // ?example=1 — бет ашылғанда test жиынынан мысалды бірден көрсету (демо үшін)
  if (randomBtn && new URLSearchParams(location.search).has("example")) request("/api/random-test");
})();
