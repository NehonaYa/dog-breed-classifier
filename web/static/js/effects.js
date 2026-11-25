// Барлық беттерге ортақ әсерлер: Vanta фоны, бет ашылғандағы анимация, скролл кезінде пайда болу,
// сандарды санау және графиктерді үлкейту. Кітапхана жоқ болса немесе пайдаланушы анимацияны
// шектесе (prefers-reduced-motion), бет анимациясыз, толық көрінеді.
(() => {
  const gsap = window.gsap;
  const header = document.querySelector(".site-header");

  // Шапкаға скролл кезінде көлеңке
  const onScroll = () => header && header.classList.toggle("is-scrolled", window.scrollY > 8);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  setupLightbox();
  setupVanta();
  if (!gsap) return;
  if (window.ScrollTrigger) gsap.registerPlugin(window.ScrollTrigger);

  const mm = gsap.matchMedia();
  mm.add("(prefers-reduced-motion: no-preference)", () => {
    // 1) Бірінші экран: элементтер кезекпен пайда болады
    const heroItems = gsap.utils.toArray("[data-hero]");
    if (heroItems.length) {
      gsap.from(heroItems, { autoAlpha: 0, y: 26, duration: 0.8, stagger: 0.09, ease: "power3.out", delay: 0.1 });
      gsap.from(".hero-visual", { autoAlpha: 0, x: 40, duration: 1, ease: "power3.out", delay: 0.35 });
      gsap.to("[data-float]", { y: -12, duration: 3.2, repeat: -1, yoyo: true, ease: "sine.inOut" });
      gsap.to("[data-float-alt]", { y: 10, duration: 2.6, repeat: -1, yoyo: true, ease: "sine.inOut", delay: 0.6 });
      gsap.from(".mock-bar i", { scaleX: 0, transformOrigin: "left center", duration: 1.2, stagger: 0.12,
                                 ease: "power3.out", delay: 0.9 });
    }

    // 2) Скролл кезінде пайда болу (бір рет)
    const reveal = gsap.utils.toArray("[data-reveal]");
    if (reveal.length && window.ScrollTrigger) {
      gsap.set(reveal, { autoAlpha: 0, y: 28 });
      ScrollTrigger.batch(reveal, {
        start: "top 90%",
        once: true,
        onEnter: (batch) => gsap.to(batch, { autoAlpha: 1, y: 0, duration: 0.7, stagger: 0.08, ease: "power3.out",
                                             overwrite: true }),
      });
    }

    // 3) Сандарды 0-ден мәніне дейін санау
    gsap.utils.toArray("[data-count]").forEach((el) => {
      const end = parseFloat(el.dataset.count);
      if (!isFinite(end) || end === 0) return;
      const dec = parseInt(el.dataset.decimals || "0", 10);
      const suffix = el.dataset.suffix || "";
      const obj = { v: 0 };
      const write = () => { el.textContent = `${obj.v.toFixed(dec)}${suffix}`; };
      const tween = gsap.to(obj, { v: end, duration: 1.6, ease: "power2.out", onUpdate: write, paused: true });
      write();
      if (el.closest(".hero-dark") || !window.ScrollTrigger) {
        tween.delay(0.4).play();
      } else {
        ScrollTrigger.create({ trigger: el, start: "top 92%", once: true, onEnter: () => tween.play() });
      }
    });
  });

  // Vanta: тек басты бетте, WebGL болса және анимация шектелмесе; экраннан шыққанда өшіреміз (GPU үнемдеу)
  function setupVanta() {
    const el = document.getElementById("hero-vanta");
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!el || !window.VANTA || !window.THREE || reduce || window.innerWidth < 700 || !webglOk()) return;
    let effect = null;
    let disabled = false;
    const start = () => {
      if (effect || disabled) return;
      try {
        effect = window.VANTA.NET({
          el, THREE: window.THREE, mouseControls: true, touchControls: true, gyroControls: false,
          minHeight: 200, minWidth: 200, scale: 1, scaleMobile: 1,
          color: 0x3987e5, backgroundColor: 0x0b1630, points: 9, maxDistance: 20, spacing: 18, showDots: true,
        });
        el.classList.add("is-live");
        checkSpeed();
      } catch { effect = null; }  // фон статикалық градиент күйінде қалады
    };
    const stop = () => { if (effect) { effect.destroy(); effect = null; el.classList.remove("is-live"); } };
    // Әлсіз компьютерде (FPS < 20) анимацияны өшіріп, статикалық фонды қалдырамыз — бет баяуламауы керек
    function checkSpeed() {
      if (!document.hasFocus() || document.visibilityState !== "visible") return; // терезе көрінбесе өлшемейміз
      let frames = 0;
      const t0 = performance.now();
      const tick = () => {
        frames += 1;
        const dt = performance.now() - t0;
        if (dt < 2000) { requestAnimationFrame(tick); return; }
        if (frames / (dt / 1000) < 20 && document.hasFocus()) { disabled = true; stop(); }
      };
      setTimeout(() => requestAnimationFrame(tick), 800);
    }
    new IntersectionObserver(([entry]) => (entry.isIntersecting ? start() : stop()), { threshold: 0 }).observe(el);
  }

  function webglOk() {
    try {
      const c = document.createElement("canvas");
      return !!(c.getContext("webgl") || c.getContext("experimental-webgl"));
    } catch { return false; }
  }

  // Графикті басқанда үлкейтіп көрсету
  function setupLightbox() {
    const dlg = document.getElementById("lightbox");
    if (!dlg || typeof dlg.showModal !== "function") return;
    const img = dlg.querySelector("img");
    document.querySelectorAll("figure.fig img").forEach((fig) => {
      fig.tabIndex = 0;
      fig.setAttribute("role", "button");
      const open = () => {
        img.src = fig.currentSrc || fig.src;
        img.alt = fig.alt;
        dlg.showModal();
        if (window.gsap && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
          window.gsap.fromTo(img, { scale: 0.94, autoAlpha: 0 }, { scale: 1, autoAlpha: 1, duration: 0.3, ease: "power2.out" });
        }
      };
      fig.addEventListener("click", open);
      fig.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
    });
    dlg.addEventListener("click", (e) => { if (e.target === dlg || e.target.closest(".lightbox-close")) dlg.close(); });
  }
})();
