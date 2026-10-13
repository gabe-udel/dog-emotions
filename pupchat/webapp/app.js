/* Pupchat front end.
 *
 * Talks only to this machine's own API. The Gemini key lives on the server and is never
 * sent to the browser - /api/health reports whether a key is configured, never what it
 * is.
 *
 * Chats and dog profiles live in localStorage. Videos do NOT: a few seconds of H.264 is
 * megabytes and localStorage gives about 5, so the bytes stay in the server's upload
 * cache and the browser keeps only the hash the server returned. That also means a
 * clip survives a cache clear, and an already-measured clip is never re-measured.
 */
const API = "";                                   // same origin
const LS_CHATS = "pupchat.chats.v1";
const LS_DOGS = "pupchat.dogs.v1";
const LS_ACTIVE = "pupchat.active.v1";

let state = {
  chats: load(LS_CHATS, []),
  dogs: load(LS_DOGS, []),
  activeDogId: localStorage.getItem(LS_ACTIVE) || null,
  chatId: null,
  attach: null,          // {name, hash, duration}
  busy: false,
  limits: { min_seconds: 5, max_seconds: 10, max_bytes: 80e6 },
  editingDogId: null,
};

/* ---------------------------------------------------------------- storage */
function load(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; }
  catch { return fallback; }                      // corrupt or blocked storage
}
function save(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); }
  catch (e) { console.warn("could not save", key, e); }
}
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
const $ = (id) => document.getElementById(id);

/* ---------------------------------------------------------------- chats */
function currentChat() { return state.chats.find(c => c.id === state.chatId) || null; }

function newChat() {
  state.chatId = null;
  state.attach = null;
  renderAttach();
  renderMessages();
  renderChats();
  $("composer").focus();
}

function ensureChat(firstQuestion) {
  if (state.chatId) return currentChat();
  const chat = {
    id: uid(),
    title: firstQuestion.slice(0, 48) + (firstQuestion.length > 48 ? "…" : ""),
    dogId: state.activeDogId,
    hash: state.attach ? state.attach.hash : null,
    videoName: state.attach ? state.attach.name : null,
    messages: [],
    history: null,
    updated: Date.now(),
  };
  state.chats.unshift(chat);
  state.chatId = chat.id;
  return chat;
}

function openChat(id) {
  state.chatId = id;
  const c = currentChat();
  state.attach = c && c.hash ? { name: c.videoName, hash: c.hash, ready: true } : null;
  if (c && c.dogId) setActiveDog(c.dogId, false);
  renderAttach();
  renderMessages();
  renderChats();
  showView("chat");
}

/* ---------------------------------------------------------------- dogs */
function activeDog() { return state.dogs.find(d => d.id === state.activeDogId) || null; }

function setActiveDog(id, rerender = true) {
  state.activeDogId = id;
  localStorage.setItem(LS_ACTIVE, id || "");
  if (rerender) { renderDogs(); renderHeader(); }
  else renderHeader();
}

function newDog() {
  const dog = { id: uid(), name: "New dog", breed: "", age: "", health: "",
                training: "", triggers: "", notes: "", baselines: [] };
  state.dogs.push(dog);
  save(LS_DOGS, state.dogs);
  setActiveDog(dog.id);
  editDog(dog.id);
}

/** The profile text sent to Gemini. Plain prose, because it goes into the prompt as
 *  context - the system prompt already says a profile may arrive. Empty fields are
 *  omitted rather than sent blank, so the model is never told "Health: " and left to
 *  guess whether that means healthy or unknown. */
function profileText(dog) {
  if (!dog) return "";
  const bits = [];
  const head = [dog.name, dog.breed, dog.age].filter(Boolean).join(", ");
  if (head) bits.push(`Dog profile: ${head}.`);
  if (dog.health) bits.push(`Health: ${dog.health}`);
  if (dog.training) bits.push(`Training history: ${dog.training}`);
  if (dog.triggers) bits.push(`Known triggers: ${dog.triggers}`);
  if (dog.notes) bits.push(`Owner notes: ${dog.notes}`);
  if (dog.baselines && dog.baselines.length)
    bits.push(`${dog.baselines.length} baseline clip(s) of this dog have been recorded.`);
  return bits.length ? bits.join("\n") : "";
}

/* ---------------------------------------------------------------- rendering */
function renderHeader() {
  const d = activeDog();
  $("activeDogName").textContent = d ? d.name : "no dog selected";
}

function renderDogs() {
  const el = $("dogList");
  el.innerHTML = "";
  if (!state.dogs.length) {
    el.innerHTML = `<p class="px-3 py-2 font-body-sm text-body-sm text-outline">No dogs yet — press Add.</p>`;
    return;
  }
  for (const d of state.dogs) {
    const on = d.id === state.activeDogId;
    const a = document.createElement("a");
    a.href = "#";
    a.className = "flex items-center gap-space-sm p-2 rounded-xl border transition-all group " +
      (on ? "bg-surface-container-lowest border-outline-variant shadow-[0_1px_3px_rgba(43,41,38,0.04)]"
          : "text-on-surface-variant hover:bg-surface-container border-transparent");
    a.innerHTML =
      `<div class="w-8 h-8 rounded-full bg-secondary-container flex items-center justify-center font-label-md text-label-md font-semibold ring-1 ring-outline-variant/60">${esc((d.name || "?")[0].toUpperCase())}</div>
       <div class="flex-1 min-w-0 text-left">
         <p class="font-label-md text-label-md truncate">${esc(d.name || "Unnamed")}</p>
         <p class="font-body-sm text-body-sm text-outline truncate">${esc([d.breed, d.age].filter(Boolean).join(" · ") || "no details yet")}</p>
       </div>
       <span class="material-symbols-outlined text-[18px] text-outline opacity-0 group-hover:opacity-100">edit</span>`;
    a.onclick = (e) => { e.preventDefault(); setActiveDog(d.id); };
    a.querySelector("span.material-symbols-outlined").onclick = (e) => {
      e.preventDefault(); e.stopPropagation(); editDog(d.id);
    };
    el.appendChild(a);
  }
}

function renderChats() {
  const q = ($("search").value || "").toLowerCase().trim();
  const el = $("chatList");
  el.innerHTML = "";
  const matches = state.chats.filter(c => {
    if (!q) return true;
    const hay = (c.title + " " + c.messages.map(m => m.text).join(" ")).toLowerCase();
    return hay.includes(q);            // searches the conversation, not just the title
  });
  $("noChats").style.display = matches.length ? "none" : "block";
  $("noChats").textContent = state.chats.length && !matches.length
    ? `No chats match "${q}".` : "No chats yet.";

  for (const c of matches) {
    const on = c.id === state.chatId;
    const a = document.createElement("a");
    a.href = "#";
    a.className = "flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all group border " +
      (on ? "bg-surface-container-lowest border-outline-variant shadow-[0_1px_3px_rgba(43,41,38,0.04)] font-medium"
          : "text-on-surface-variant hover:bg-surface-container border-transparent");
    a.innerHTML =
      `<span class="material-symbols-outlined text-[18px] text-outline">chat_bubble_outline</span>
       <span class="font-body-sm text-body-sm truncate flex-1">${esc(c.title)}</span>
       <span class="material-symbols-outlined text-[16px] text-outline opacity-0 group-hover:opacity-100 hover:text-error" data-del="1">delete</span>`;
    a.onclick = (e) => {
      e.preventDefault();
      if (e.target.dataset.del) {
        state.chats = state.chats.filter(x => x.id !== c.id);
        if (state.chatId === c.id) newChat();
        save(LS_CHATS, state.chats); renderChats();
        return;
      }
      openChat(c.id);
    };
    el.appendChild(a);
  }
}

function renderMessages() {
  const c = currentChat();
  const box = $("messages");
  box.innerHTML = "";
  $("emptyState").style.display = (c && c.messages.length) ? "none" : "block";
  if (!c) return;
  for (const m of c.messages) box.appendChild(bubble(m));
  window.scrollTo(0, document.body.scrollHeight);
}

function bubble(m) {
  const wrap = document.createElement("div");
  if (m.role === "user") {
    wrap.className = "flex justify-end";
    wrap.innerHTML =
      `<div class="max-w-[85%] space-y-space-sm">
         ${m.videoName ? `<div class="flex items-center gap-2 justify-end font-label-sm text-label-sm text-outline">
            <span class="material-symbols-outlined text-[16px]">movie</span>${esc(m.videoName)}</div>` : ""}
         <div class="px-4 py-3 rounded-2xl bg-primary text-on-primary font-body-md text-body-md">${esc(m.text)}</div>
       </div>`;
  } else if (m.role === "error") {
    wrap.className = "flex";
    wrap.innerHTML = `<div class="max-w-[85%] px-4 py-3 rounded-2xl bg-error-container text-on-error-container font-body-sm text-body-sm">${esc(m.text)}</div>`;
  } else {
    wrap.className = "flex";
    wrap.innerHTML =
      `<div class="max-w-[90%] space-y-space-sm">
         <div class="flex items-center gap-2">
           <span class="material-symbols-outlined text-[18px] text-primary">pets</span>
           <span class="font-label-md text-label-md font-semibold">Pupchat</span>
           ${m.meta ? `<span class="font-label-sm text-label-sm text-outline">${esc(m.meta)}</span>` : ""}
         </div>
         <div class="px-4 py-3 rounded-2xl bg-surface-container-low border border-outline-variant/60 font-body-md text-body-md md">${mdToHtml(m.text)}</div>
       </div>`;
  }
  return wrap;
}

/** Minimal markdown: the answer format is headings, bullets and numbered steps. Escapes
 *  first, so model output can never inject markup. */
function mdToHtml(s) {
  let t = esc(s);
  t = t.replace(/^\s*\*\*(.+?)\*\*\s*$/gm, "<h3>$1</h3>");
  t = t.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  t = t.replace(/^\s*[-*]\s+(.*)$/gm, "<li>$1</li>");
  t = t.replace(/(<li>[\s\S]*?<\/li>)(?!\s*<li>)/g, "<ul>$1</ul>");
  return t;
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g,
    ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

function renderAttach() {
  const row = $("attachRow");
  if (!state.attach) { row.classList.add("hidden"); return; }
  row.classList.remove("hidden");
  $("attachName").textContent = state.attach.name;
  $("attachState").textContent = state.attach.ready ? "measured" : "measuring…";
}

/* ---------------------------------------------------------------- views */
function showView(which) {
  $("chatView").classList.toggle("hidden", which !== "chat");
  $("profileView").classList.toggle("hidden", which !== "profile");
}

/* ---------------------------------------------------------------- upload */
async function pickVideo(file, { baseline = false } = {}) {
  if (!file) return null;
  const lim = state.limits;
  if (file.size > lim.max_bytes) {
    toastError(`That file is ${(file.size / 1e6).toFixed(0)} MB; the limit is ${(lim.max_bytes / 1e6).toFixed(0)} MB.`);
    return null;
  }
  // Check length in the browser first - instant, and saves uploading a clip the
  // server is going to reject anyway. The server checks again; this is convenience.
  const dur = await durationOf(file).catch(() => null);
  if (dur !== null && (dur < lim.min_seconds - 0.25 || dur > lim.max_seconds + 0.25)) {
    toastError(`That clip is ${dur.toFixed(1)}s. Pupchat needs ${lim.min_seconds}–${lim.max_seconds} seconds.`);
    return null;
  }

  if (!baseline) {
    state.attach = { name: file.name, hash: null, ready: false };
    renderAttach();
  }
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API}/api/analyse`, { method: "POST", body: fd });
  if (!r.ok) {
    const msg = (await r.json().catch(() => ({}))).detail || `upload failed (${r.status})`;
    toastError(msg);
    if (!baseline) { state.attach = null; renderAttach(); }
    return null;
  }
  const data = await r.json();
  if (!baseline) {
    state.attach = { name: file.name, hash: data.hash, ready: true,
                     duration: data.duration_s };
    renderAttach();
  }
  return { name: file.name, hash: data.hash, duration: data.duration_s };
}

function durationOf(file) {
  return new Promise((resolve, reject) => {
    const v = document.createElement("video");
    v.preload = "metadata";
    v.onloadedmetadata = () => { URL.revokeObjectURL(v.src); resolve(v.duration); };
    v.onerror = () => { URL.revokeObjectURL(v.src); reject(new Error("unreadable")); };
    v.src = URL.createObjectURL(file);
  });
}

function toastError(text) {
  const c = ensureChat("(error)");
  c.messages.push({ role: "error", text });
  save(LS_CHATS, state.chats);
  renderMessages(); renderChats();
}

/* ---------------------------------------------------------------- send */
async function send() {
  if (state.busy) return;
  const text = $("composer").value.trim();
  if (!text) return;

  if (!state.chatId && !state.attach) {
    toastError("Add a 5–10 second clip with the + button before the first question.");
    return;
  }
  if (state.attach && !state.attach.ready) {
    toastError("The clip is still being measured — give it a moment.");
    return;
  }

  const chat = ensureChat(text);
  if (!chat.hash && state.attach) { chat.hash = state.attach.hash; chat.videoName = state.attach.name; }
  if (!chat.dogId) chat.dogId = state.activeDogId;

  chat.messages.push({ role: "user", text,
                       videoName: chat.messages.length ? null : chat.videoName });
  chat.updated = Date.now();
  $("composer").value = "";
  autoGrow();
  save(LS_CHATS, state.chats);
  renderMessages(); renderChats();

  state.busy = true;
  $("sendBtn").disabled = true;
  const thinking = document.createElement("div");
  thinking.className = "flex";
  thinking.innerHTML =
    `<div class="px-4 py-3 rounded-2xl bg-surface-container-low border border-outline-variant/60 font-body-sm text-body-sm text-outline">
       <span class="dot">●</span><span class="dot">●</span><span class="dot">●</span>
       <span class="ml-2">${chat.messages.length <= 1 ? "measuring the video, this takes a few minutes" : "thinking"}</span>
     </div>`;
  $("messages").appendChild(thinking);
  window.scrollTo(0, document.body.scrollHeight);

  try {
    const fd = new FormData();
    fd.append("question", text);
    fd.append("hash", chat.hash);
    if (chat.history) fd.append("history", JSON.stringify(chat.history));
    const prof = profileText(state.dogs.find(d => d.id === chat.dogId));
    if (prof) fd.append("profile", prof);

    const r = await fetch(`${API}/api/ask`, { method: "POST", body: fd });
    thinking.remove();
    if (!r.ok) {
      const msg = (await r.json().catch(() => ({}))).detail || `request failed (${r.status})`;
      chat.messages.push({ role: "error", text: msg });
    } else {
      const d = await r.json();
      chat.history = d.history;
      const cost = (d.cost_usd || d.cost_usd === 0) ? `· $${Number(d.cost_usd).toFixed(4)}` : "";
      chat.messages.push({ role: "assistant", text: d.answer,
                           meta: `${d.model} · ${Math.round(d.seconds)}s ${cost}` });
    }
  } catch (e) {
    thinking.remove();
    chat.messages.push({ role: "error", text: `could not reach Pupchat: ${e.message}` });
  } finally {
    state.busy = false;
    $("sendBtn").disabled = false;
    chat.updated = Date.now();
    save(LS_CHATS, state.chats);
    renderMessages(); renderChats();
  }
}

/* ---------------------------------------------------------------- profile edit */
function editDog(id) {
  state.editingDogId = id;
  const d = state.dogs.find(x => x.id === id);
  if (!d) return;
  $("profileTitle").textContent = d.name || "Dog profile";
  for (const [k, el] of Object.entries(fields())) el.value = d[k] || "";
  renderBaselines();
  showView("profile");
}

function fields() {
  return { name: $("p_name"), breed: $("p_breed"), age: $("p_age"),
           triggers: $("p_triggers"), health: $("p_health"),
           training: $("p_training"), notes: $("p_notes") };
}

function saveDogEdits() {
  const d = state.dogs.find(x => x.id === state.editingDogId);
  if (!d) return;
  for (const [k, el] of Object.entries(fields())) d[k] = el.value;
  $("profileTitle").textContent = d.name || "Dog profile";
  save(LS_DOGS, state.dogs);
  renderDogs(); renderHeader();
}

function renderBaselines() {
  const d = state.dogs.find(x => x.id === state.editingDogId);
  const el = $("baselineList");
  el.innerHTML = "";
  if (!d) return;
  (d.baselines || []).forEach((b, i) => {
    const card = document.createElement("div");
    card.className = "flex items-center gap-space-sm px-3 py-2 rounded-xl bg-surface-container border border-outline-variant";
    card.innerHTML =
      `<span class="material-symbols-outlined text-[18px] text-primary">movie</span>
       <span class="font-body-sm text-body-sm truncate flex-1">${esc(b.name)}</span>
       <span class="font-label-sm text-label-sm text-outline">${b.duration ? b.duration.toFixed(1) + "s" : ""}</span>
       <button class="material-symbols-outlined text-[18px] text-outline hover:text-error">close</button>`;
    card.querySelector("button").onclick = () => {
      d.baselines.splice(i, 1); save(LS_DOGS, state.dogs); renderBaselines();
    };
    el.appendChild(card);
  });
}

/* ---------------------------------------------------------------- wiring */
function autoGrow() {
  const t = $("composer");
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 160) + "px";
}

async function boot() {
  try {
    const h = await (await fetch(`${API}/api/health`)).json();
    state.limits = h.upload || state.limits;
    $("healthDot").className = "w-2 h-2 rounded-full " + (h.key_configured ? "bg-primary" : "bg-error");
    $("healthText").textContent = h.key_configured
      ? `${h.provider} · ready` : `${h.provider} · no API key set`;
    $("modelBadge").textContent = h.model || "";
  } catch {
    $("healthDot").className = "w-2 h-2 rounded-full bg-error";
    $("healthText").textContent = "server not reachable";
  }

  $("newChat").onclick = newChat;
  $("addDog").onclick = newDog;
  $("search").oninput = renderChats;
  $("plusBtn").onclick = () => $("fileInput").click();
  $("fileInput").onchange = (e) => { pickVideo(e.target.files[0]); e.target.value = ""; };
  $("attachClear").onclick = () => { state.attach = null; renderAttach(); };
  $("sendBtn").onclick = send;
  $("composer").oninput = autoGrow;
  $("composer").onkeydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };
  $("openProfile").onclick = () => {
    if (state.activeDogId) editDog(state.activeDogId); else newDog();
  };
  $("closeProfile").onclick = () => { saveDogEdits(); showView("chat"); };
  $("deleteDog").onclick = () => {
    state.dogs = state.dogs.filter(d => d.id !== state.editingDogId);
    if (state.activeDogId === state.editingDogId) setActiveDog(state.dogs[0]?.id || null);
    save(LS_DOGS, state.dogs); renderDogs(); renderHeader(); showView("chat");
  };
  for (const el of Object.values(fields())) el.oninput = saveDogEdits;
  $("addBaseline").onclick = () => $("baselineInput").click();
  $("baselineInput").onchange = async (e) => {
    const f = e.target.files[0]; e.target.value = "";
    const got = await pickVideo(f, { baseline: true });
    if (!got) return;
    const d = state.dogs.find(x => x.id === state.editingDogId);
    d.baselines = d.baselines || [];
    d.baselines.push(got);
    save(LS_DOGS, state.dogs);
    renderBaselines();
  };

  if (!state.activeDogId && state.dogs.length) setActiveDog(state.dogs[0].id, false);
  renderDogs(); renderChats(); renderHeader(); renderMessages(); renderAttach();
}

boot();
