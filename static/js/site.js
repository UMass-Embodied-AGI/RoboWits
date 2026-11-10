/* RoboWits project page — interactions.
   Data comes from static/js/data.js (generated from the repo). */
(function () {
  "use strict";

  var RW = window.RW;
  var $  = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var VID = "static/videos/";
  var PRE = "static/videos/preview/";   /* 640px hover encodes, ~13x smaller */
  var POS = "static/images/posters/";
  var POLICIES = [
    { key: "act",  label: "ACT" },
    { key: "pi0",  label: "π₀" },
    { key: "pi05", label: "π₀.₅" },
    { key: "astra", label: "Astra-Policy" },
    { key: "fable", label: "Fable 5.2" }
  ];
  /* Astra-Policy and Fable 5.2 ran on smaller episode budgets, and Fable has
     gaps (cells it never ran are null), so missing values render as a dash.
     Fable's Full-suite averages cover only the tasks it ran. */
  /* The gallery is clips only — tasks without a recorded clip are left out
     rather than represented by a still. The results table still covers all 30. */
  var CLIPS = RW.tasks.filter(function (t) { return !!t.video; });
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function el(tag, attrs, kids) {
    var n = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === "class") n.className = attrs[k];
      else if (k === "html") n.innerHTML = attrs[k];
      else if (k === "text") n.textContent = attrs[k];
      else if (attrs[k] != null) n.setAttribute(k, attrs[k]);
    });
    (kids || []).forEach(function (c) { if (c) n.appendChild(c); });
    return n;
  }
  function svgIcon(id, cls) {
    var s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    if (cls) s.setAttribute("class", cls);
    s.setAttribute("aria-hidden", "true");
    var u = document.createElementNS("http://www.w3.org/2000/svg", "use");
    u.setAttribute("href", "#" + id);
    s.appendChild(u);
    return s;
  }
  var esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); };

  /* ------------------------------------------------------------------ theme */
  var toggle = $("#themeToggle");
  toggle.addEventListener("click", function () {
    var cur = document.documentElement.getAttribute("data-theme");
    if (!cur) cur = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    var next = cur === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("rw-theme", next); } catch (e) {}
    if (chartState) drawResults();
  });

  /* ------------------------------------------------- sticky nav + scrollspy */
  var nav = $("#nav");
  var onScroll = function () { nav.classList.toggle("is-stuck", window.scrollY > 8); };
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });

  var navLinks = $$(".nav__links a");
  var sections = navLinks.map(function (a) { return $(a.getAttribute("href")); }).filter(Boolean);
  if ("IntersectionObserver" in window && sections.length) {
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        navLinks.forEach(function (a) {
          a.setAttribute("aria-current", a.getAttribute("href") === "#" + e.target.id ? "true" : "false");
        });
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    sections.forEach(function (s) { spy.observe(s); });
  }

  /* ----------------------------------------------------------- copy buttons */
  document.addEventListener("click", function (ev) {
    var btn = ev.target.closest("[data-copy]");
    if (!btn) return;
    var src = $(btn.getAttribute("data-copy"));
    if (!src) return;
    var done = function () {
      btn.classList.add("is-done");
      var lbl = $("span", btn), was = lbl.textContent;
      lbl.textContent = "Copied";
      $("use", btn).setAttribute("href", "#i-check");
      setTimeout(function () {
        btn.classList.remove("is-done");
        lbl.textContent = was;
        $("use", btn).setAttribute("href", "#i-copy");
      }, 1800);
    };
    if (navigator.clipboard) navigator.clipboard.writeText(src.textContent).then(done, function () {});
    else {
      var r = document.createRange(); r.selectNodeContents(src);
      var sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
      try { document.execCommand("copy"); done(); } catch (e) {}
    }
  });

  /* ================================================================ GALLERY */
  var gallery = $("#gallery");
  var visible = [];

  function tileFor(t, idx) {
    var media = el("div", { class: "tile__media" }, [
      el("img", { src: POS + t.poster, alt: "", width: "720", height: "405", loading: idx < 8 ? "eager" : "lazy", decoding: "async" })
    ]);
    var flags = el("div", { class: "tile__flags" });
    if (t.core10) flags.appendChild(el("span", { class: "tile__flag", text: "RW-10" }));
    if (flags.childNodes.length) media.appendChild(flags);
    var play = el("span", { class: "tile__play" });
    play.appendChild(svgIcon("i-play"));
    media.appendChild(play);

    var bar = el("span", { class: "tile__bar" });
    bar.appendChild(el("i", { style: "width:" + Math.round(t.mutations / 8 * 100) + "%" }));

    var tile = el("button", {
      class: "tile", type: "button", "data-id": t.id,
      "aria-label": t.name + " — " + t.instruction
    }, [
      media,
      el("div", { class: "tile__body" }, [
        el("div", { class: "tile__name", text: t.name }),
        el("div", { class: "tile__instr", text: t.instruction }),
        el("div", { class: "tile__meta" }, [
          el("span", { text: t.mutations + " mutations" }), bar
        ])
      ])
    ]);

    if (!reduceMotion) {
      var v = null;
      var start = function () {
        if (v) { v.play().catch(function () {}); return; }
        media.classList.add("is-loading");
        v = el("video", { muted: "", loop: "", playsinline: "", preload: "none", "aria-hidden": "true", tabindex: "-1" });
        v.muted = true;
        v.src = PRE + t.video;
        media.appendChild(v);
        /* Reveal as soon as a frame can be painted — waiting for "playing"
           leaves the tile looking dead while the clip buffers. */
        var reveal = function () { media.classList.remove("is-loading"); v.classList.add("is-on"); };
        v.addEventListener("loadeddata", reveal);
        v.addEventListener("playing", reveal);
        v.addEventListener("error", function () { media.classList.remove("is-loading"); });
        v.play().catch(function () {});
      };
      var stop = function () {
        media.classList.remove("is-loading");
        if (v) { v.classList.remove("is-on"); v.pause(); }
      };
      tile.addEventListener("pointerenter", start);
      tile.addEventListener("focus", start);
      tile.addEventListener("pointerleave", stop);
      tile.addEventListener("blur", stop);
    }

    tile.addEventListener("click", function () { openLightbox(t.id); });
    return tile;
  }

  function renderGallery() {
    visible = CLIPS.slice();
    gallery.textContent = "";
    visible.forEach(function (t, i) { gallery.appendChild(tileFor(t, i)); });
  }
  renderGallery();

  /* =============================================================== LIGHTBOX */
  var lb = $("#lightbox");

  function pct(v) { return v.toFixed(1) + "%"; }
  var DASH = '<span class="zero">—</span>';

  function taskDetail(t) {
    var head = "<tr><th>Policy</th><th>Seed SR</th><th>Seed PS</th><th>Mut SR</th><th>Mut PS</th></tr>";
    var rows = POLICIES.map(function (p) {
      var r = t.results[p.key];
      var c = function (x) { return x ? "<td>" + pct(x[0]) + "</td><td>" + x[1].toFixed(2) + "</td>" : "<td>" + DASH + "</td><td>" + DASH + "</td>"; };
      return "<tr><td>" + p.label + "</td>" + c(r.seed) + c(r.mut) + "</tr>";
    }).join("");

    var media = '<video controls autoplay muted loop playsinline preload="auto" poster="' + POS + t.poster + '">' +
                '<source src="' + VID + t.video + '" type="video/mp4"></video>';

    var chips = ['<span class="chip">' + t.mutations + " mutated variants</span>"];
    if (t.core10) chips.push('<span class="chip chip--core">RoboWits-10</span>');
    if (t.altVideo) chips.push('<button class="chip" type="button" data-alt="' + t.altVideo + '">Second view</button>');

    var idx = visible.indexOf(t);
    var pager = idx < 0 ? "" :
      '<div class="lb__nav">' +
        '<button class="icon-btn" type="button" data-step="-1" aria-label="Previous task"><svg aria-hidden="true"><use href="#i-left"/></svg></button>' +
        '<button class="icon-btn" type="button" data-step="1" aria-label="Next task"><svg aria-hidden="true"><use href="#i-right"/></svg></button>' +
        '<span class="count">' + (idx + 1) + " / " + visible.length + "</span>" +
      "</div>";

    return '' +
      '<div class="lb__card">' +
        '<div class="lb__media">' + media + "</div>" +
        '<div class="lb__body">' +
          '<div class="lb__top">' +
            "<div><h3>" + esc(t.name) + "</h3></div>" +
            '<button class="icon-btn lb__close" type="button" data-close aria-label="Close"><svg aria-hidden="true"><use href="#i-close"/></svg></button>' +
          "</div>" +
          '<p class="lb__instr">“' + esc(t.instruction) + '”</p>' +
          '<div class="lb__chips">' + chips.join("") + "</div>" +
          '<table class="mini"><thead>' + head + "</thead><tbody>" + rows + "</tbody></table>" +
          pager +
        "</div>" +
      "</div>";
  }

  function openLightbox(id) {
    var t = CLIPS.filter(function (x) { return x.id === id; })[0];
    if (!t) return;
    lb.innerHTML = taskDetail(t);
    lb.dataset.id = id;
    if (!lb.open) lb.showModal();
    var v = $("video", lb);
    if (v) v.play().catch(function () {});
  }

  function stepLightbox(d) {
    var i = visible.findIndex(function (x) { return x.id === lb.dataset.id; });
    if (i < 0) return;
    openLightbox(visible[(i + d + visible.length) % visible.length].id);
  }

  lb.addEventListener("click", function (ev) {
    if (ev.target === lb) { lb.close(); return; }
    var close = ev.target.closest("[data-close]");
    if (close) { lb.close(); return; }
    var step = ev.target.closest("[data-step]");
    if (step) { stepLightbox(Number(step.getAttribute("data-step"))); return; }
    var alt = ev.target.closest("[data-alt]");
    if (alt) {
      var v = $("video", lb);
      if (v) { v.src = VID + alt.getAttribute("data-alt"); v.play().catch(function () {}); }
      alt.remove();
    }
  });
  lb.addEventListener("close", function () { lb.innerHTML = ""; });
  document.addEventListener("keydown", function (ev) {
    if (!lb.open) return;
    if (ev.key === "ArrowRight") { ev.preventDefault(); stepLightbox(1); }
    if (ev.key === "ArrowLeft")  { ev.preventDefault(); stepLightbox(-1); }
  });

  /* zoomable figures */
  $$(".zoom").forEach(function (img) {
    img.addEventListener("click", function () {
      lb.innerHTML = '<div class="lb__card">' +
        '<div class="lb__media" style="background:var(--surface)"><img src="' + img.getAttribute("src") +
        '" alt="' + esc(img.getAttribute("alt")) + '" style="aspect-ratio:auto;background:var(--surface)"></div>' +
        '<div class="lb__body"><div class="lb__top"><div><h3>' + esc(img.dataset.zoomCaption || "Figure") + "</h3></div>" +
        '<button class="icon-btn lb__close" type="button" data-close aria-label="Close"><svg aria-hidden="true"><use href="#i-close"/></svg></button>' +
        "</div></div></div>";
      lb.dataset.id = "";
      lb.showModal();
    });
  });

  /* ====================================================== MUTATION EXAMPLES */
  var mutTabs = $("#mutTabs"), mutPanel = $("#mutPanel");

  function renderMut(i) {
    var m = RW.mutationPairs[i];
    $$("button", mutTabs).forEach(function (b, j) { b.setAttribute("aria-selected", String(i === j)); });
    var trip = el("div", { class: "triptych" });
    m.clips.forEach(function (c) {
      var isSeed = c.kind === "seed";
      var poster = POS + c.v.replace(".mp4", ".webp");
      var card = el("figure", { class: "clip " + (isSeed ? "clip--seed" : "clip--mut"), style: "margin:0" });
      var v = el("video", { controls: "", muted: "", loop: "", playsinline: "", preload: "none", poster: poster });
      v.muted = true;
      v.appendChild(el("source", { src: VID + c.v, type: "video/mp4" }));
      card.appendChild(v);
      var kindChip = isSeed
        ? '<span class="chip chip--seed"><i class="chip__dot"></i>Seed</span>'
        : '<span class="chip chip--mut"><i class="chip__dot"></i>Mutation</span>';
      card.appendChild(el("figcaption", { class: "clip__body", html:
        '<div style="display:flex;align-items:center;gap:8px;justify-content:space-between">' +
        "<h4>" + esc(c.label) + "</h4>" + kindChip + "</div>" +
        "<p>" + esc(c.note) + "</p>" }));
      trip.appendChild(card);
    });
    mutPanel.textContent = "";
    mutPanel.appendChild(el("div", { class: "instr-line", html:
      "<b>Instruction</b><span>“" + esc(m.instruction) + "” &mdash; identical across all three scenes.</span>" }));
    mutPanel.appendChild(el("div", { style: "height:14px" }));
    mutPanel.appendChild(trip);
  }

  RW.mutationPairs.forEach(function (m, i) {
    var b = el("button", { class: "tab", type: "button", role: "tab", "aria-controls": "mutPanel",
      "aria-selected": String(i === 0),
      text: m.name });
    b.addEventListener("click", function () { renderMut(i); });
    mutTabs.appendChild(b);
  });
  renderMut(0);

  /* ================================================================ RESULTS */
  var chartState = { suite: "rw10" };   /* both metrics are drawn side by side */

  function rowsForSuite() {
    return chartState.suite === "rw10"
      ? RW.tasks.filter(function (t) { return t.core10; })
      : RW.tasks.slice();
  }
  function val(res, cond, metric) { return metric === "sr" ? res[cond][0] : res[cond][1]; }
  function fmt(v, metric) { return metric === "sr" ? v.toFixed(1) + "%" : v.toFixed(2); }

  /* --- grouped column chart ------------------------------------------------ */
  var tip = $("#tip");
  function showTip(x, y, html) {
    tip.innerHTML = html;
    tip.style.left = x + "px";
    tip.style.top = y + "px";
    tip.classList.add("is-on");
  }
  function hideTip() { tip.classList.remove("is-on"); }

  function drawChart(metric, hostId) {
    var avg = RW.averages[chartState.suite];
    var series = [
      { cond: "seed", label: "Seed task",        color: "var(--seed)" },
      { cond: "mut",  label: "Mutated variants", color: "var(--mut)" }
    ];
    var data = POLICIES.filter(function (p) { return avg[p.key].seed && avg[p.key].mut; }).map(function (p) {
      return { label: p.label, key: p.key, seed: avg[p.key].seed, mut: avg[p.key].mut };
    });

    var W = 470, H = 300, PL = 44, PR = 12, PT = 24, PB = 48;
    var pw = W - PL - PR, ph = H - PT - PB;
    var maxV = metric === "sr"
      ? Math.max.apply(null, data.map(function (d) { return Math.max(val(d, "seed", metric), val(d, "mut", metric)); }))
      : 1;
    var top = metric === "sr" ? Math.ceil(maxV / 5) * 5 + 5 : 1;
    var ticks = metric === "sr" ? [0, top / 2, top] : [0, 0.25, 0.5, 0.75, 1];

    var y = function (v) { return PT + ph - (v / top) * ph; };
    var gw = pw / data.length;
    var BW = 22, GAP = 2.5;

    var parts = [];
    ticks.forEach(function (t) {
      parts.push('<line class="grid-line" x1="' + PL + '" x2="' + (W - PR) + '" y1="' + y(t) + '" y2="' + y(t) + '"/>');
      parts.push('<text class="axis-text" x="' + (PL - 10) + '" y="' + (y(t) + 4) + '" text-anchor="end">' +
        (metric === "sr" ? t.toFixed(0) + "%" : t.toFixed(2)) + "</text>");
    });

    data.forEach(function (d, gi) {
      var cx = PL + gw * gi + gw / 2;
      var x0 = cx - BW - GAP / 2;
      series.forEach(function (s, si) {
        var v = val(d, s.cond, metric);
        var bx = x0 + si * (BW + GAP);
        var bh = Math.max((v / top) * ph, v > 0 ? 2 : 0);
        var by = PT + ph - bh;
        var r = Math.min(3.5, bh);
        if (bh > 0) {
          parts.push('<path class="bar" data-k="' + hostId + "-" + d.key + "-" + s.cond + '" fill="' + s.color + '" d="' +
            "M" + bx + " " + (PT + ph) + " L" + bx + " " + (by + r) +
            " Q" + bx + " " + by + " " + (bx + r) + " " + by +
            " L" + (bx + BW - r) + " " + by +
            " Q" + (bx + BW) + " " + by + " " + (bx + BW) + " " + (by + r) +
            " L" + (bx + BW) + " " + (PT + ph) + " Z" + '"/>');
        }
        parts.push('<text class="value-text" x="' + (bx + BW / 2) + '" y="' + (by - 7) + '" text-anchor="middle">' + fmt(v, metric) + "</text>");
        parts.push('<rect class="bar-hit" data-k="' + hostId + "-" + d.key + "-" + s.cond +
          '" data-pol="' + d.label + '" data-cond="' + s.label +
          '" data-sr="' + d[s.cond][0].toFixed(1) + '" data-ps="' + d[s.cond][1].toFixed(2) +
          '" x="' + (bx - 4) + '" y="' + PT + '" width="' + (BW + 8) + '" height="' + ph + '"/>');
      });
      parts.push('<text class="group-text" x="' + cx + '" y="' + (PT + ph + 24) + '" text-anchor="middle">' + d.label + "</text>");
    });

    parts.push('<line class="grid-line" x1="' + PL + '" x2="' + (W - PR) + '" y1="' + (PT + ph) + '" y2="' + (PT + ph) + '"/>');

    $("#" + hostId).innerHTML =
      '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' +
      (metric === "sr" ? "Average success rate" : "Average progress score") +
      ' by policy on seed tasks and their mutated variants. The table below lists every value.">' +
      parts.join("") + "</svg>";

    var chart = $("#" + hostId);
    $$(".bar-hit", chart).forEach(function (hit) {
      hit.addEventListener("pointerenter", function (ev) {
        chart.classList.add("is-hovering");
        $$(".bar", chart).forEach(function (b) {
          b.classList.toggle("is-active", b.getAttribute("data-k") === hit.getAttribute("data-k"));
        });
        var r = hit.getBoundingClientRect();
        showTip(r.left + r.width / 2, r.top + 40,
          "<b>" + hit.dataset.pol + " · " + hit.dataset.cond + "</b>" +
          "<dl><dt>Success rate</dt><dd>" + hit.dataset.sr + "%</dd>" +
          "<dt>Progress score</dt><dd>" + hit.dataset.ps + "</dd></dl>");
      });
      hit.addEventListener("pointerleave", function () {
        chart.classList.remove("is-hovering");
        $$(".bar", chart).forEach(function (b) { b.classList.remove("is-active"); });
        hideTip();
      });
    });

    $(metric === "sr" ? "#chartSubSR" : "#chartSubPS").textContent = "Mean over the " +
      (chartState.suite === "rw10" ? "10 RoboWits-10 tasks"
        : "30 RoboWits-Full tasks · Fable 5.2: seed over 28 tasks, mutations over 10");
  }

  /* --- per-task table ------------------------------------------------------ */
  function sr(x) { return x ? x[0] : -1; }

  function drawTable() {
    var rows = rowsForSuite();
    var avg = RW.averages[chartState.suite];
    var maxV = Math.max.apply(null, rows.map(function (t) {
      return Math.max.apply(null, POLICIES.map(function (p) {
        return Math.max(sr(t.results[p.key].seed), sr(t.results[p.key].mut)); })); }));

    var head =
      '<thead><tr class="pol"><th class="task"></th>' +
        POLICIES.map(function (p) { return '<th colspan="2">' + p.label + "</th>"; }).join("") +
      "</tr><tr><th class='task'>Task</th>" +
        POLICIES.map(function (p, i) {
          return '<th' + (i ? ' class="sep"' : "") + ">Seed</th><th>Mut</th>"; }).join("") +
      "</tr></thead>";

    /* Each cell carries both measures — success rate in ink, progress score
       muted beside it — since there is no longer a metric switch. The wash
       encodes the success rate only. */
    function cells(r, isAvg) {
      var bestSeed = Math.max.apply(null, POLICIES.map(function (p) { return sr(r[p.key].seed); }));
      var bestMut  = Math.max.apply(null, POLICIES.map(function (p) { return sr(r[p.key].mut); }));
      return POLICIES.map(function (p, i) {
        return ["seed", "mut"].map(function (cond, j) {
          var cell = r[p.key][cond];
          if (!cell) return '<td class="v' + (i && !j ? " sep" : "") + '"><span class="num">' + DASH + "</span></td>";
          var sr = cell[0], ps = cell[1];
          var best = !isAvg && sr > 0 && sr === (cond === "seed" ? bestSeed : bestMut);
          var w = Math.max(0, Math.min(1, sr / maxV)) * 100;
          var color = cond === "seed" ? "var(--seed-wash)" : "var(--mut-wash)";
          return '<td class="v' + (best ? " is-best" : "") + (i && !j ? " sep" : "") + '">' +
            (sr > 0 ? '<span class="wash" style="width:' + Math.max(w, 2.5).toFixed(1) + "%;background:" + color + '"></span>' : "") +
            '<span class="num' + (sr === 0 ? " zero" : "") + '">' + sr.toFixed(1) + "%" +
            '<i class="ps">' + ps.toFixed(2) + "</i></span></td>";
        }).join("");
      }).join("");
    }

    var body = rows.map(function (t) {
      return '<tr data-id="' + t.id + '"><td class="task">' + esc(t.name) + "</td>" +
        cells(t.results, false) + "</tr>";
    }).join("");

    var avgRow = '<tr class="is-avg"><td class="task">Average</td>' + cells(avg, true) + "</tr>";

    $("#resultTable").innerHTML = head + "<tbody>" + body + avgRow + "</tbody>";
    $("#tableTitle").textContent = "Per-task results";
    $("#tableSub").textContent = chartState.suite === "rw10"
      ? "RoboWits-10 · success rate and progress score · 50 episodes per seed task, 20 per mutation (Astra-Policy 10 / 5, Fable 5.2 5 / 1-3)"
      : "RoboWits-Full · success rate and progress score · all 30 seed tasks and their mutation families · Fable 5.2 averages only the tasks it ran: seed on 28 (not 15 and 22), mutations on the 10 RoboWits-10 tasks";

    $$("#resultTable tbody tr[data-id]").forEach(function (tr) {
      var id = tr.getAttribute("data-id");
      if (!CLIPS.some(function (x) { return x.id === id; })) return;
      tr.style.cursor = "pointer";
      tr.title = "Watch this task";
      tr.addEventListener("click", function () { openLightbox(id); });
    });
  }

  function drawResults() { drawChart("sr", "chartSR"); drawChart("ps", "chartPS"); drawTable(); }

  $$("[data-suite]").forEach(function (b) {
    b.addEventListener("click", function () {
      chartState.suite = b.getAttribute("data-suite");
      $$("[data-suite]").forEach(function (o) { o.setAttribute("aria-pressed", String(o === b)); });
      drawResults();
    });
  });
  drawResults();

  /* hero montage: attach the source after first paint; a still poster is
     already in place, and readers who asked for less motion keep it. */
  (function () {
    var hero = $("#heroVideo");
    if (!hero) return;
    if (reduceMotion) { hero.removeAttribute("autoplay"); hero.setAttribute("controls", ""); }
    var attach = function () {
      hero.src = VID + "card.mp4";
      if (!reduceMotion) hero.play().catch(function () {});
    };
    if (document.readyState === "complete") attach();
    else window.addEventListener("load", attach);
  })();

  /* Poster-session strip: counts down, flips to "happening now" during the
     session and to past tense afterwards, so the page stays truthful on its
     own once the conference is over. All comparisons are in absolute time. */
  (function () {
    var box = $("#posterEvent");
    if (!box) return;
    var start = Date.parse(box.dataset.start), end = Date.parse(box.dataset.end);
    if (isNaN(start) || isNaN(end)) return;

    var badge = $("#eventWhen"), now = Date.now();
    if (now > end) {
      box.classList.add("is-past");
      $("#eventLabel").textContent = "Presented at NeurIPS 2026";
      $("#eventDetail").textContent = "Hall C1, Atlanta";
      var ics = $("#eventIcs");
      if (ics) ics.hidden = true;
      return;
    }
    if (now >= start) {
      box.classList.add("is-now");
      badge.textContent = "Happening now";
    } else {
      var hours = (start - now) / 3600000;
      badge.textContent = hours < 24
        ? "in " + Math.max(1, Math.round(hours)) + (Math.round(hours) === 1 ? " hour" : " hours")
        : "in " + Math.round(hours / 24) + " days";
    }
    badge.hidden = false;
  })();

  /* The merged pipeline clip carries the whole method, so it gets a scrub bar:
     play/pause, seek, and a tick where the right-hand panels take over. It
     auto-plays when scrolled into view, but never overrides a manual pause. */
  (function () {
    var v = $("#pipelineVideo");
    if (!v) return;
    var wrap = v.closest(".vplayer"),
        btn = $("#pipePlay"), seek = $("#pipeSeek"),
        time = $("#pipeTime"), mark = $("#pipeMark");
    var PHASE_TWO = 2.4;          // seconds: the right-hand panels start here
    var userPaused = reduceMotion, scrubbing = false, loaded = false;

    function fmt(s) {
      if (!isFinite(s)) return "0:00";
      var m = Math.floor(s / 60), r = Math.floor(s % 60);
      return m + ":" + (r < 10 ? "0" : "") + r;
    }
    function ensure() {
      if (loaded) return;
      loaded = true; v.preload = "auto"; v.load();
    }
    function sync() {
      if (scrubbing || !v.duration) return;
      seek.value = String(Math.round(v.currentTime / v.duration * 1000));
      time.textContent = fmt(v.currentTime);
    }

    v.addEventListener("loadedmetadata", function () {
      time.textContent = fmt(0);
      if (v.duration) {
        mark.style.left = (PHASE_TWO / v.duration * 100) + "%";
        mark.hidden = false;
      }
    });
    v.addEventListener("timeupdate", sync);
    v.addEventListener("play", function () { wrap.classList.add("is-playing"); });
    v.addEventListener("pause", function () { wrap.classList.remove("is-playing"); });

    function toggle() {
      ensure();
      if (v.paused) { userPaused = false; v.play().catch(function () {}); }
      else { userPaused = true; v.pause(); }
      btn.setAttribute("aria-label", v.paused ? "Play" : "Pause");
    }
    btn.addEventListener("click", toggle);
    v.addEventListener("click", toggle);

    seek.addEventListener("pointerdown", function () { scrubbing = true; });
    ["pointerup", "pointercancel"].forEach(function (e) {
      seek.addEventListener(e, function () { scrubbing = false; });
    });
    seek.addEventListener("input", function () {
      ensure();
      if (!v.duration) return;
      v.currentTime = seek.value / 1000 * v.duration;
      time.textContent = fmt(v.currentTime);
    });

    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) {
            if (userPaused) return;
            ensure();
            v.play().catch(function () {});
          } else if (!v.paused) {
            v.pause();
          }
        });
      }, { threshold: 0.25 });
      io.observe(v);
    } else { ensure(); }
  })();

  /* ============================================================== STEPPER */
  var STAGES = [
    { k: "01", icon: "seed",   title: "Seed task generator",
      text: "Proposes candidate tasks as text: a name, an object list, an instruction and a potential solution." },
    { k: "02", icon: "verify", title: "Task verifier",
      text: "Rejects candidates that are trivial, unsolvable or physically incoherent, and feeds the reason back." },
    { k: "03", icon: "metric", title: "Metric generator",
      text: "Writes the success criteria as checkable Python over object state \u2014 not a learned or subjective score." },
    { k: "04", icon: "mutate", title: "Task mutator",
      text: "Pivots, traps and additions turn one verified seed into a family of graded variants, each re-verified." },
    { k: "05", icon: "scene",  title: "Scene generator",
      text: "Instantiates every verified task as randomised 3D scene configurations ready to execute." }
  ];

  (function () {
    var dots = $("#stepperDots"), panel = $("#stepperPanel");
    if (!dots || !panel) return;
    var buttons = [];

    function show(i) {
      buttons.forEach(function (b, j) { b.setAttribute("aria-selected", String(i === j)); });
      var s = STAGES[i];
      panel.classList.add("is-swapping");
      setTimeout(function () {
        panel.innerHTML =
          '<div class="stepper__icon"><img src="static/images/agents/' + s.icon +
          '.png" alt="" width="272" height="272" loading="lazy"></div>' +
          '<div><span class="stepper__k">AGENT ' + s.k + "</span><h3>" + esc(s.title) +
          "</h3><p>" + esc(s.text) + "</p></div>";
        panel.classList.remove("is-swapping");
      }, reduceMotion ? 0 : 140);
    }

    STAGES.forEach(function (s, i) {
      var b = el("button", { class: "stepper__dot", type: "button", role: "tab",
        "aria-selected": String(i === 0), "aria-controls": "stepperPanel",
        "aria-label": "Agent " + s.k + ": " + s.title, text: s.k });
      b.addEventListener("click", function () { show(i); });
      b.addEventListener("keydown", function (ev) {
        var d = ev.key === "ArrowRight" ? 1 : ev.key === "ArrowLeft" ? -1 : 0;
        if (!d) return;
        ev.preventDefault();
        var n = (i + d + STAGES.length) % STAGES.length;
        buttons[n].focus(); show(n);
      });
      buttons.push(b);
      dots.appendChild(el("li", {}, [b]));
    });
    show(0);
  })();
})();
