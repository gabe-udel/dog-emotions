/* Pupchat - entry point.
 *
 * The page is the Google Stitch design, merged into one document: one shell, three
 * screen bodies tagged `data-view`, and a set of <template>s cut from the mockup's own
 * rows. Nothing here draws anything the design did not already contain.
 *
 * No API key ever reaches the browser. The page talks only to this server, which holds
 * the key in .env and never passes it on.
 */

import { api, ApiError } from "./api.js";
import * as chat from "./chat.js";
import * as router from "./router.js";
import { wireShare } from "./share.js";
import * as store from "./store.js";
import { state } from "./store.js";
import * as views from "./views.js";
import { wireVideoModal } from "./video.js";
import { $, viewRoot } from "./views.js";

/* ------------------------------------------------------------------ views */

/* The Tailwind `hidden` CLASS, never the HTML attribute: two of the three view roots
 * carry display utilities, and `display:flex` beats the user agent's `[hidden]` rule -
 * so the attribute would leave every view visible at once. */
export function showView(name) {
  for (const el of document.querySelectorAll("[data-view]")) {
    const on = el.dataset.view === name;
    el.classList.toggle("hidden", !on);
    if (on && el.dataset.view === "chat") el.classList.add("flex");
  }
  closeDrawer();
}

/* ------------------------------------------------------------------ sidebar */

function openDrawer() {
  document.querySelector("[data-sidebar]")?.classList.remove("-translate-x-full");
  document.querySelector("[data-scrim]")?.classList.remove("hidden");
}

export function closeDrawer() {
  document.querySelector("[data-sidebar]")?.classList.add("-translate-x-full");
  document.querySelector("[data-scrim]")?.classList.add("hidden");
}

/* ------------------------------------------------------------------ errors */

let toastTimer = null;

/* Errors are transient UI, not conversation.
 *
 * The previous page routed them through the chat store, so every rejected upload
 * created a chat titled "(error)" containing only that error. The sidebar filled with
 * empty chats, which reads exactly like history not being saved. An error is feedback
 * about something that did not happen; it has no place in a transcript. */
export function toastError(text) {
  const el = $("toast");
  if (!el) return;
  $("toastText").textContent = text;
  el.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), 7000);
}

/* A blank screen is never an acceptable failure, so anything unhandled becomes a
 * message the owner can read. */
window.addEventListener("error", (e) => toastError(`Something went wrong: ${e.message}`));
window.addEventListener("unhandledrejection", (e) =>
  toastError(e.reason instanceof ApiError
    ? e.reason.message
    : `Something went wrong: ${e.reason?.message || e.reason}`));

/* ------------------------------------------------------------------ credits */

/* Credits and, on a shared server, what a visitor should know before uploading.
 *
 * Both are rendered from /api/health rather than written into the HTML, so the licence
 * text, the retention window and the daily caps have exactly one source. A page that
 * promised "deleted after 24 hours" while the server ran a different window would be
 * worse than one that promised nothing. */
export function showCredits(h) {
  if (h.attribution) $("creditLine").textContent = h.attribution;

  const share = h.share || {};
  if (!share.enabled) return;
  const el = $("shareNotice");
  el.textContent =
    `Shared demo: videos you upload are deleted after ${share.retention_hours} hours. ` +
    `${share.videos_per_day} videos and ${share.asks_per_day} questions per visitor per day. ` +
    `Please don't upload anything you wouldn't want a stranger to see.`;
  el.classList.remove("hidden");
}

/* ------------------------------------------------------------------ routing */

async function applyRoute(route) {
  state.chatId = route.view === "chat" ? route.chatId : null;

  if (route.view === "dog") {
    if (route.isNew) {
      // Created on arrival rather than on first keystroke, so the autosave below has
      // somewhere to save to and the URL can name the dog straight away.
      const dog = await store.addDog({ name: "" });
      views.renderDogPage(dog.id, { isNew: false });
      router.go(`/dogs/${encodeURIComponent(store.slug(dog.name) || dog.id)}`,
                { replace: true });
      showView("dog");
      renderAll();
      viewRoot("dog")?.querySelector('[data-field="name"]')?.focus();
      return;
    }
    if (!store.dogByKey(route.dogKey)) {
      toastError("That dog is not here any more.");
      router.go("/", { replace: true });
      return;
    }
    views.renderDogPage(route.dogKey);
  }

  if (route.view === "chat") {
    if (state.ready && !store.chatById(route.chatId)) {
      toastError("That chat is not here any more.");
      router.go("/", { replace: true });
      return;
    }
    showView("chat");
    renderAll();
    await chat.renderChat(route.chatId);
    return;
  }

  showView(route.view);
  renderAll();
}

function renderAll() {
  views.renderDogs();
  views.renderChats();
  views.renderDogContext();
  const route = router.current();
  if (route.view === "dog" && !route.isNew) views.renderDogPage(route.dogKey);
}

/* ------------------------------------------------------------------ wiring */

function wireSidebar() {
  document.querySelector("[data-menu]")?.addEventListener("click", openDrawer);
  document.querySelector("[data-scrim]")?.addEventListener("click", closeDrawer);
  document
    .querySelector('button[title="Collapse sidebar"]')
    ?.addEventListener("click", closeDrawer);

  const search = document.querySelector('aside input[type="text"]');
  search?.addEventListener("input", () => {
    state.search = search.value;
    views.renderChats();
  });

  for (const a of document.querySelectorAll('a[data-path="new-chat"]')) {
    a.setAttribute("href", "/");
  }
  for (const a of document.querySelectorAll('a[data-path="add-dog"]')) {
    a.setAttribute("href", "/dogs/new");
  }
  // The brand is a link home, which the export drew as a bare image.
  const brand = document.querySelector('aside img[alt="Pupchat Brand Logo"]')?.parentElement;
  if (brand && brand.tagName !== "A") {
    brand.style.cursor = "pointer";
    brand.addEventListener("click", () => router.go("/"));
  }
}

function wireDogMenu() {
  let open = false;
  const trigger = document.querySelector("header button");
  trigger?.addEventListener("click", (e) => {
    e.stopPropagation();
    open = !open;
    views.renderDogMenu(open);
  });
  document.addEventListener("click", () => {
    if (open) {
      open = false;
      views.renderDogMenu(false);
    }
  });
}

/* Autosave, debounced.
 *
 * Saving on every keystroke would be a request per character; saving only on blur
 * loses the last field whenever someone closes the tab while still in it. Both. */
function wireDogEditing() {
  const root = viewRoot("dog");
  if (!root) return;
  const timers = new Map();

  const save = async (key, value) => {
    const id = views.editingDog();
    if (!id) return;
    try {
      await store.saveDog(id, { [key]: value });
      if (key === "name") {
        const dog = store.dogById(id);
        const path = `/dogs/${encodeURIComponent(store.slug(dog?.name) || id)}`;
        if (location.pathname !== path) history.replaceState({}, "", path);
      }
    } catch (e) {
      toastError(e.message);
    }
  };

  for (const el of root.querySelectorAll("input[data-field], textarea[data-field]")) {
    const key = el.dataset.field;
    el.addEventListener("input", () => {
      clearTimeout(timers.get(key));
      timers.set(key, setTimeout(() => save(key, el.value), 500));
    });
    el.addEventListener("blur", () => {
      clearTimeout(timers.get(key));
      save(key, el.value);
    });
  }

  const avatar = root.querySelector('[data-avatar="dog"]');
  const picker = root.querySelector('input[data-input="photo"]');
  avatar?.addEventListener("click", () => picker?.click());
  picker?.addEventListener("change", async () => {
    const file = picker.files?.[0];
    picker.value = "";
    if (!file || !views.editingDog()) return;
    try {
      await api.uploadPhoto(views.editingDog(), file);
      await store.refresh();
    } catch (e) {
      toastError(e.message);
    }
  });
}

/* ------------------------------------------------------------------ composer */

/* Both screens have one - the welcome screen's textarea and the chat screen's input.
 * They are wired together so a question typed on either goes the same way. */
function composers() {
  return [
    { box: $("prompt-input"), send: $("send-button"),
      file: $("media-upload-input") },
    { box: viewRoot("chat")?.querySelector('input[type="text"]'),
      send: viewRoot("chat")?.querySelector("button[title='Send message']"),
      file: null },
  ].filter((c) => c.box);
}

function wireComposers() {
  const fileInput = $("media-upload-input");

  const submit = async (box) => {
    const text = box.value;
    box.value = "";
    autoGrow(box);
    await chat.send(text, {
      onError: toastError,
      onChat: (id) => {
        // replace, not push: the question is not a place you navigate back to.
        history.replaceState({}, "", `/chat/${id}`);
        showView("chat");
        renderAll();
      },
    });
    renderAll();
  };

  for (const c of composers()) {
    c.send?.addEventListener("click", () => submit(c.box));
    c.box.addEventListener("input", () => autoGrow(c.box));
    c.box.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        submit(c.box);
      }
    });
  }

  // Every way of reaching the file picker: the + on the welcome screen, the + on the
  // chat screen, and the "Upload a clip" item in its popover.
  for (const el of document.querySelectorAll(
    '#toggle-popover-btn, [data-action="upload"], [title="Upload canine video or photo"]'
  )) {
    el.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      fileInput?.click();
      $("media-upload-popover")?.classList.add("hidden");
    });
  }
  fileInput?.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    fileInput.value = "";
    await chat.attachFile(file, toastError);
  });
  document.querySelector('#media-chip button')?.addEventListener("click", () =>
    chat.setAttachment(null));

  for (const card of document.querySelectorAll("[data-prompt]")) {
    card.addEventListener("click", () => {
      const box = composers()[0]?.box;
      if (!box) return;
      box.value = card.dataset.prompt;
      autoGrow(box);
      box.focus();
    });
  }
}

function autoGrow(box) {
  if (box.tagName !== "TEXTAREA") return;
  box.style.height = "auto";
  box.style.height = Math.min(box.scrollHeight, 144) + "px";
}

/* ------------------------------------------------------------------ boot */

async function boot() {
  wireSidebar();
  wireDogMenu();
  wireDogEditing();
  wireComposers();
  wireVideoModal();
  wireShare();
  $("toastClose")?.addEventListener("click", () => $("toast").classList.add("hidden"));
  store.subscribe(renderAll);

  try {
    const h = await api.health();
    state.health = h;
    $("healthDot").className =
      "w-2 h-2 rounded-full shrink-0 " + (h.key_configured ? "bg-primary" : "bg-error");
    $("healthText").textContent = h.key_configured
      ? `${h.provider} · ready`
      : `${h.provider} · no API key set`;
    showCredits(h);
  } catch {
    $("healthDot").className = "w-2 h-2 rounded-full shrink-0 bg-error";
    $("healthText").textContent = "server not reachable";
  }

  try {
    await store.refresh();
  } catch (e) {
    toastError(e.message);
  }

  if (!state.owner) {
    // A visitor on a shared demo. Say so once, where the chat list would be, rather
    // than leaving them wondering why the sidebar is empty.
    const empty = document.querySelector('[data-empty="chats"]');
    if (empty) empty.textContent = "Chats aren't saved on the shared demo.";
    const dogs = document.querySelector('[data-empty="dogs"]');
    if (dogs) dogs.textContent = "Dogs are private on a shared demo.";
    for (const a of document.querySelectorAll('a[data-path="add-dog"]')) {
      a.classList.add("hidden");
    }
  }

  router.onRoute((route) => {
    applyRoute(route).catch((e) => toastError(e.message));
  });
  router.start();
}

/* A module script is deferred, so DOMContentLoaded has usually already fired by the
 * time this runs. Waiting for it would mean waiting forever. */
boot();
