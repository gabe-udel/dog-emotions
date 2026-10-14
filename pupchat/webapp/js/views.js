/* Drawing the page from the store.
 *
 * Every repeated row is a <template> cut from the Stitch export, so the look is the
 * design's and this file only decides what text goes where. Nothing here builds markup
 * from strings - `tpl()` clones, `fill()` writes text into `data-field` nodes.
 */

import { api } from "./api.js";
import * as store from "./store.js";
import { state, visibleChats, dogById, activeDog, chatsForDog } from "./store.js";

export const $ = (id) => document.getElementById(id);
export const viewRoot = (name) => document.querySelector(`[data-view="${name}"]`);

/* ------------------------------------------------------------------ helpers */

export function tpl(name) {
  const t = document.getElementById(`tpl-${name}`);
  if (!t) throw new Error(`missing template: tpl-${name}`);
  return t.content.firstElementChild.cloneNode(true);
}

/* Writes into [data-field] nodes. textContent, never innerHTML: a dog's name is typed
 * by a person and ends up in the sidebar, the header and the page title, so it must
 * never be parsed as markup. */
export function fill(root, values) {
  for (const [key, value] of Object.entries(values)) {
    for (const el of root.querySelectorAll(`[data-field="${key}"]`)) {
      el.textContent = value == null ? "" : String(value);
    }
  }
}

export const initial = (name) => (name || "?").trim().charAt(0).toUpperCase() || "?";

/* A photo if there is one, the first letter if not. Same element either way, so a dog
 * without a picture looks deliberate rather than broken. */
export function paintAvatar(el, dog) {
  if (!el) return;
  const url = api.photoUrl(dog);
  const letter = el.querySelector('[data-field="dogInitial"]');
  if (letter) letter.textContent = initial(dog?.name);
  el.style.backgroundImage = url ? `url("${url}")` : "";
  if (letter) letter.classList.toggle("hidden", Boolean(url));
}

export const dogMeta = (dog) =>
  [dog?.breed, dog?.age].filter(Boolean).join(" · ") || "no details yet";

function setEmpty(key, show) {
  const el = document.querySelector(`[data-empty="${key}"]`);
  if (el) el.classList.toggle("hidden", !show);
}

/* ------------------------------------------------------------------ sidebar */

export function renderDogs() {
  const nav = document.querySelectorAll("nav[data-active-classes]")[0];
  if (!nav) return;
  const activeClasses = (nav.dataset.activeClasses || "").split(/\s+/).filter(Boolean);

  for (const el of nav.querySelectorAll("a")) el.remove();
  setEmpty("dogs", state.dogs.length === 0);

  for (const dog of state.dogs) {
    const row = tpl("dog-row");
    row.setAttribute("href", `/dogs/${encodeURIComponent(store.slug(dog.name))}`);
    fill(row, { dogName: dog.name || "Unnamed" });
    const name = row.querySelector("p");
    if (name) name.textContent = dog.name || "Unnamed";
    const sub = row.querySelectorAll("p")[1];
    if (sub) sub.textContent = dogMeta(dog);
    const badge = row.querySelector("div");
    if (badge) {
      badge.textContent = "";
      badge.classList.add("bg-cover", "bg-center");
      const span = document.createElement("span");
      span.dataset.field = "dogInitial";
      badge.appendChild(span);
      paintAvatar(badge, dog);
    }
    if (dog.id === state.activeDogId) {
      row.classList.add(...activeClasses);
      row.setAttribute("aria-current", "page");
    }
    nav.appendChild(row);
  }
}

export function renderChats() {
  const nav = document.querySelectorAll("nav[data-active-classes]")[1];
  if (!nav) return;
  const activeClasses = (nav.dataset.activeClasses || "").split(/\s+/).filter(Boolean);

  for (const el of nav.querySelectorAll("a")) el.remove();
  const chats = visibleChats();
  setEmpty("chats", chats.length === 0);

  const empty = document.querySelector('[data-empty="chats"]');
  if (empty && state.chats.length && !chats.length) {
    empty.textContent = `No chats match “${state.search}”.`;
  } else if (empty) {
    empty.textContent = "No chats yet. Add a clip and ask a question.";
  }

  for (const chat of chats) {
    const row = tpl("chat-row");
    row.setAttribute("href", `/chat/${chat.id}`);
    const label = row.querySelector("span.truncate, span.flex-1") ||
      row.querySelectorAll("span")[1];
    if (label) label.textContent = chat.title || "New chat";
    if (chat.id === state.chatId) {
      row.classList.add(...activeClasses);
      row.setAttribute("aria-current", "page");
    }
    nav.appendChild(row);
  }
}

/* ------------------------------------------------------------------ the header */

export function renderDogContext() {
  const dog = activeDog();
  for (const root of [document.body]) {
    fill(root, {
      dogName: dog?.name || "no dog yet",
      dogMeta: dog ? dogMeta(dog) : "",
      baselineCount: dog?.baselines?.length ?? 0,
    });
  }
  for (const el of document.querySelectorAll('[data-avatar="dog"]')) {
    paintAvatar(el, dog);
  }

  const strip = document.querySelector("[data-baseline-strip]");
  if (strip) {
    const n = dog?.baselines?.length ?? 0;
    strip.classList.toggle("hidden", n === 0);
    strip.classList.toggle("flex", n > 0);
  }

  for (const link of document.querySelectorAll('[data-link="dog"], a[data-path="dog-profile"]')) {
    link.setAttribute(
      "href",
      dog ? `/dogs/${encodeURIComponent(store.slug(dog.name))}` : "/dogs/new"
    );
  }
}

/* The dog switcher. Built here rather than in the export, which drew a chevron with no
 * menu behind it. */
export function renderDogMenu(open) {
  let menu = document.getElementById("dogMenu");
  const anchor = document.querySelector("header .relative.inline-flex") ||
    document.querySelector("header button")?.parentElement;
  if (!anchor) return;
  if (!menu) {
    menu = document.createElement("div");
    menu.id = "dogMenu";
    menu.className =
      "hidden absolute top-full left-0 mt-1 w-64 z-40 rounded-xl bg-surface-container-lowest " +
      "border border-outline-variant shadow-lg p-1.5 space-y-0.5";
    anchor.classList.add("relative");
    anchor.appendChild(menu);
  }
  menu.classList.toggle("hidden", !open);
  if (!open) return;

  menu.textContent = "";
  if (!state.dogs.length) {
    const p = document.createElement("p");
    p.className = "px-3 py-2 font-body-sm text-body-sm text-outline";
    p.textContent = "No dogs yet.";
    menu.appendChild(p);
  }
  for (const dog of state.dogs) {
    const b = document.createElement("button");
    b.type = "button";
    b.className =
      "w-full text-left px-3 py-2 rounded-lg font-label-md text-label-md " +
      "hover:bg-surface-container flex items-center justify-between gap-2";
    b.textContent = dog.name || "Unnamed";
    if (dog.id === state.activeDogId) {
      const tick = document.createElement("span");
      tick.className = "material-symbols-outlined text-[16px] text-primary";
      tick.textContent = "check";
      b.appendChild(tick);
    }
    b.addEventListener("click", async () => {
      await store.chooseDog(dog.id);
      renderDogMenu(false);
    });
    menu.appendChild(b);
  }
  const add = document.createElement("a");
  add.href = "/dogs/new";
  add.className =
    "block px-3 py-2 rounded-lg font-label-md text-label-md text-primary hover:bg-surface-container";
  add.textContent = "+ Add a dog";
  menu.appendChild(add);
}

/* ------------------------------------------------------------------ dog profile */

let editingDogId = null;

export function renderDogPage(dogKey, { isNew = false } = {}) {
  const root = viewRoot("dog");
  if (!root) return;

  const dog = isNew ? null : store.dogByKey(dogKey);
  editingDogId = dog?.id ?? null;

  fill(root, {
    dogName: dog?.name || (isNew ? "New dog" : "Not found"),
    dogMeta: dog ? dogMeta(dog) : "Fill in whatever you know.",
    statAge: dog?.age || "—",
    statBreed: dog?.breed || "—",
    statBaselines: dog?.baselines?.length ?? 0,
    statChats: dog ? chatsForDog(dog.id).length : 0,
    baselineCount: `${dog?.baselines?.length ?? 0} saved`,
  });
  paintAvatar(root.querySelector('[data-avatar="dog"]'), dog);

  for (const el of root.querySelectorAll("[data-field]")) {
    const key = el.dataset.field;
    if (el.tagName === "INPUT" || el.tagName === "TEXTAREA") {
      el.value = dog?.[key] ?? "";
      el.disabled = false;
    }
  }
  renderBaselines(root, dog);
  renderPastChats(root, dog);
}

function renderBaselines(root, dog) {
  const grid = root.querySelector("[data-baselines]");
  if (!grid) return;
  for (const el of grid.querySelectorAll("[data-baseline]")) el.remove();

  const addTile = grid.querySelector("button");
  for (const b of dog?.baselines || []) {
    const tile = tpl("baseline-tile");
    tile.setAttribute("data-baseline", b.hash);
    const thumb = tile.querySelector("[data-thumb]");
    if (thumb) {
      const v = document.createElement("video");
      v.src = `${api.videoUrl(b.hash)}#t=0.1`;
      v.muted = true;
      v.loop = true;
      v.playsInline = true;
      v.preload = "metadata";
      v.className = "w-full h-full object-cover";
      thumb.textContent = "";
      thumb.appendChild(v);
    }
    const texts = tile.querySelectorAll("p");
    if (texts[0]) texts[0].textContent = b.name || "clip";
    if (texts[1]) {
      texts[1].textContent = b.duration ? `${Number(b.duration).toFixed(1)}s` : "";
    }
    const badge = tile.querySelector("span.absolute");
    if (badge) badge.textContent = b.duration ? `${Math.round(b.duration)}s` : "";
    grid.insertBefore(tile, addTile || null);
  }
}

function renderPastChats(root, dog) {
  const container = root.querySelector("[data-past-chats]");
  if (!container) return;
  for (const el of container.querySelectorAll("[data-past-chat]")) el.remove();

  const chats = dog ? chatsForDog(dog.id) : [];
  const empty = container.querySelector("[data-empty-chats]");
  empty?.classList.toggle("hidden", chats.length > 0);

  for (const chat of chats.slice(0, 6)) {
    const row = tpl("past-chat-row");
    row.setAttribute("data-past-chat", chat.id);
    row.setAttribute("href", `/chat/${chat.id}`);
    const ps = row.querySelectorAll("p");
    if (ps[0]) ps[0].textContent = chat.title || "New chat";
    if (ps[1]) ps[1].textContent = `${chat.n_messages || 0} messages`;
    container.appendChild(row);
  }
}

export const editingDog = () => editingDogId;
