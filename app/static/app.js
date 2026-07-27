/* AdForge Studio — single-page flow. Every section is always visible; status
   chips + skeletons make it obvious what is happening and where results land. */
const $ = (s) => document.querySelector(s);
const key = () => $("#apiKey").value.trim();

function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg; t.className = "show" + (err ? " err" : "");
  setTimeout(() => (t.className = ""), 4500);
}
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body && typeof opts.body !== "string" ? JSON.stringify(opts.body) : opts.body,
  });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}
function follow(jobId, onEvent) {
  return new Promise((resolve, reject) => {
    const es = new EventSource(`/api/jobs/${jobId}/events`);
    es.onmessage = (m) => {
      const ev = JSON.parse(m.data);
      if (ev.event === "result") {
        es.close();
        ev.status === "done" ? resolve(ev.result) : reject(new Error(ev.error || "job failed"));
      } else onEvent?.(ev);
    };
    es.onerror = () => { es.close(); reject(new Error("connection lost")); };
  });
}

/* ---------- flow state: one source of truth for section status ---------- */
const CHIP_TEXT = {
  1: { wait: "Start here", ready: "Start here", busy: "Loading…", done: "Reference loaded ✓" },
  2: { wait: "Waiting for a reference", ready: "Ready — describe your product", busy: "Directing… ~30s", done: "Blueprint ready ✓" },
  3: { wait: "Waiting for the blueprint", ready: "Ready — pick a style", busy: "Painting frames…", done: "Storyboard ready ✓" },
  4: { wait: "Waiting for the storyboard", ready: "Ready — preview cost, then render", busy: "Filming…", done: "Commercial ready ✓" },
};
function setStatus(n, status, text) {
  const sec = $(`#s${n}`);
  sec.dataset.status = status;
  $(`#chip${n}`).textContent = text || CHIP_TEXT[n][status];
  $(`#chip${n}`).dataset.status = status;
  const bar = document.querySelector(`.fstep[data-s="${n}"]`);
  bar.dataset.status = status;
  if (status === "ready") sec.scrollIntoView({ behavior: "smooth", block: "start" });
}
const skeletons = (n, cls) =>
  Array.from({ length: n }, () => `<div class="skeleton ${cls}"></div>`).join("");

/* ================= 1 · reference & assets ================= */
async function refLoaded(preview) {
  $("#refZone").hidden = true;
  $("#refPreview").src = preview; $("#refPreview").hidden = false;
  setStatus(1, "done");
  setStatus(2, "ready");
}
$("#btnRefUrl").onclick = async () => {
  try {
    setStatus(1, "busy", "Downloading reference…");
    const r = await api("/api/reference/url", { method: "POST", body: { url: $("#refUrl").value } });
    await refLoaded(r.preview);
  } catch (e) { setStatus(1, "ready"); toast(e.message, true); }
};
$("#refFile").onchange = async (ev) => {
  const fd = new FormData(); fd.append("file", ev.target.files[0]);
  try {
    setStatus(1, "busy", "Uploading…");
    const r = await (await fetch("/api/reference/upload", { method: "POST", body: fd })).json();
    await refLoaded(r.preview);
  } catch (e) { setStatus(1, "ready"); toast(e.message, true); }
};
$("#assetFile").onchange = async (ev) => {
  const fd = new FormData(); fd.append("file", ev.target.files[0]);
  try {
    const r = await (await fetch("/api/assets/upload", { method: "POST", body: fd })).json();
    $("#assetStrip .dropzone")?.remove();
    const d = document.createElement("div");
    d.className = "asset";
    d.innerHTML = `<img src="${r.preview}"><span class="tag">${r.tag}</span>`;
    $("#assetStrip").appendChild(d);
    toast(`${r.tag} added — the blueprint will swap it in`);
  } catch (e) { toast(e.message, true); }
};
$("#btnAssetsClear").onclick = async () => {
  await api("/api/assets", { method: "DELETE" });
  $("#assetStrip").innerHTML =
    '<div class="dropzone slim"><span>Uploaded assets appear here as Asset 1, Asset 2…</span></div>';
};

/* ================= 2 · blueprint ================= */
$("#duration").oninput = (e) => ($("#durOut").textContent = e.target.value + "s");

$("#btnAnalyze").onclick = async () => {
  if (!key()) return toast("Paste your FAL API key (top right) first", true);
  try {
    setStatus(2, "busy");
    $("#sceneList").innerHTML = skeletons(3, "scene-skel");
    const r = await api("/api/analyze", { method: "POST", body: {
      api_key: key(), directive: $("#directive").value,
      duration: +$("#duration").value, production_type: $("#prodType").value } });
    renderCampaign(r.campaign); renderScenes(r.scenes);
    setStatus(2, "done"); setStatus(3, "ready");
    toast("Blueprint ready — tweak any scene, then generate frames");
  } catch (e) {
    setStatus(2, "ready");
    $("#sceneList").innerHTML = '<div class="placeholder">Your scene-by-scene plan appears here.</div>';
    toast(e.message, true);
  }
};
function renderCampaign(c) {
  const el = $("#campaignCard");
  el.hidden = false;
  el.innerHTML = `<b>Core idea</b> ${c.core_idea}<br>
    <b>Product</b> ${c.product} · <b>Tone</b> ${c.tone} · <b>Style</b> ${c.visual_style}`;
}
function renderScenes(scenes) {
  $("#sceneList").innerHTML = "";
  scenes.forEach((s, i) => {
    const d = document.createElement("div");
    d.className = "scene";
    d.innerHTML = `
      <div class="stop"><b>Scene ${s.scene_number}</b><span class="role">${s.role_in_story || ""}</span></div>
      <div class="meta">${s.what_happens || ""}</div>
      <textarea data-i="${i}" data-f="visual" rows="2">${s.visual || ""}</textarea>
      <textarea data-i="${i}" data-f="dialogue" rows="1"
        placeholder="Spoken words (empty = silent)">${s.dialogue || ""}</textarea>
      <div class="meta">motion: ${s.motion || "—"} · transition: ${s.transition}
        · assets: ${(s.assigned_assets || []).join(", ") || "auto"}</div>`;
    $("#sceneList").appendChild(d);
  });
  window._scenes = scenes;
  $("#sceneList").oninput = (e) => {
    const t = e.target;
    if (t.dataset.f) window._scenes[+t.dataset.i][t.dataset.f] = t.value;
  };
}
const pushSceneEdits = async () =>
  window._scenes && api("/api/scenes", { method: "PUT", body: window._scenes });

/* ================= 3 · storyboard ================= */
$("#btnStoryboard").onclick = async () => {
  if (!window._scenes?.length) return toast("Direct the blueprint first (step 2)", true);
  try {
    await pushSceneEdits();
    setStatus(3, "busy");
    $("#frameGrid").innerHTML = window._scenes.map((s) =>
      `<div class="frame" id="fr-${s.scene_number}">
         <div class="spin"><div class="loader"></div>scene ${s.scene_number}</div>
         <div class="cap">${s.role_in_story || "rendering…"}</div></div>`).join("");
    const { job_id } = await api("/api/storyboard", { method: "POST", body: {
      api_key: key(), aspect_ratio: $("#aspect").value, font_style: $("#fontStyle").value,
      image_model: $("#imgModel").value } });
    const result = await follow(job_id, (ev) => {
      if (ev.event !== "frame") return;
      const cell = $(`#fr-${ev.scene_number}`); if (!cell) return;
      cell.innerHTML = ev.url
        ? `<img src="${ev.url}"><div class="cap">Scene ${ev.scene_number} ✓</div>`
        : `<div class="spin">✖</div><div class="cap">${ev.error || "failed"}</div>`;
      if (!ev.url) cell.classList.add("err");
    });
    window._scenes = result.scenes;
    setStatus(3, "done"); setStatus(4, "ready");
    toast(`Storyboard done — images ≈ $${result.image_cost.toFixed(3)}`);
  } catch (e) { setStatus(3, "ready"); toast(e.message, true); }
};

/* ================= 4 · film ================= */
$("#btnTimeline").onclick = async () => {
  try {
    const t = await api(`/api/timeline?video_model=${$("#vidModel").value}`);
    const box = $("#timeline");
    window._overrides = t.rows.map((r) => r.seconds);
    box.innerHTML = t.rows.map((r, i) => `
      <div class="trow"><b>Scene ${r.scene_number}</b>
        <input type="number" step="1" min="2" max="15" value="${Math.round(r.seconds)}" data-i="${i}">
        <span>${r.dialogue_words} words · ${r.role || ""}</span></div>`).join("")
      + `<div class="trow total">Planned ≈ ${Math.round(t.rows.reduce((a, r) => a + r.seconds, 0))}s
         (target ${t.target}s) · video ≈ $${t.estimated_cost.toFixed(2)}</div>`;
    box.oninput = (e) => { if (e.target.dataset.i) window._overrides[+e.target.dataset.i] = +e.target.value; };
  } catch (e) { toast(e.message, true); }
};

$("#btnRender").onclick = async () => {
  const log = $("#renderLog"); log.innerHTML = "";
  const line = (txt, cls = "") => { log.innerHTML += `<div class="${cls}">${txt}</div>`; log.scrollTop = 1e9; };
  try {
    setStatus(4, "busy");
    const { job_id } = await api("/api/render", { method: "POST", body: {
      api_key: key(), video_model: $("#vidModel").value, resolution: $("#resolution").value,
      aspect_ratio: $("#aspect").value, max_budget: +$("#budget").value,
      duration_overrides: window._overrides || null } });
    line("Rendering all segments in parallel — each line updates as a scene lands…");
    const result = await follow(job_id, (ev) => {
      if (ev.event === "segment")
        line(`scene ${ev.scene_number}: ${ev.status}${ev.error ? " — " + ev.error : ""}`,
             ev.status === "ok" ? "ok" : "err");
    });
    if (result.final) {
      $("#finalVideo").src = result.final; $("#finalVideo").hidden = false;
      setStatus(4, "done");
      line(`Done · video ≈ $${result.video_cost.toFixed(2)}`, "ok");
    } else setStatus(4, "ready");
    if (result.failed?.length) {
      $("#recovery").hidden = false;
      $("#fixScene").value = result.failed[0];
      line(`Failed scenes: ${result.failed.join(", ")} — fix below, the rest is kept`, "err");
    }
  } catch (e) { setStatus(4, "ready"); toast(e.message, true); }
};

$("#btnFix").onclick = async () => {
  try {
    toast("Re-rendering segment…");
    const r = await api("/api/render/segment", { method: "POST", body: {
      api_key: key(), scene_number: +$("#fixScene").value,
      prompt_override: $("#fixPrompt").value, image_override: $("#fixImage").value } });
    if (r.final) { $("#finalVideo").src = r.final + "?t=" + Date.now(); $("#finalVideo").hidden = false; }
    toast("Segment replaced and re-stitched");
  } catch (e) { toast(e.message, true); }
};

/* ================= boot: models + session restore ================= */
(async () => {
  try {
    const m = await api("/api/models");
    $("#vidModel").innerHTML = Object.entries(m.video)
      .map(([k, v]) => `<option value="${k}" ${k === m.defaults.video ? "selected" : ""}>
        ${v.label} · $${v.price_per_sec.toFixed(3)}/s</option>`).join("");
    $("#imgModel").innerHTML = Object.entries(m.image)
      .map(([k, v]) => `<option value="${k}" ${k === m.defaults.image ? "selected" : ""}>${v.label}</option>`).join("");
    const st = await api("/api/state");
    if (st.reference) { setStatus(1, "done"); setStatus(2, "ready"); }
    if (st.campaign) { renderCampaign(st.campaign); setStatus(2, "done"); setStatus(3, "ready"); }
    if (st.scenes?.length) {
      renderScenes(st.scenes);
      const withImg = st.scenes.filter((s) => s.generated_image_url);
      if (withImg.length) {
        $("#frameGrid").innerHTML = withImg.map((s) =>
          `<div class="frame"><img src="${s.generated_image_url}">
           <div class="cap">Scene ${s.scene_number}</div></div>`).join("");
        setStatus(3, "done"); setStatus(4, "ready");
        toast(`Restored previous session (${withImg.length} frames)`);
      }
    }
    if (st.final) { $("#finalVideo").src = st.final; $("#finalVideo").hidden = false; setStatus(4, "done"); }
  } catch { /* fresh session */ }
})();
