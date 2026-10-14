/* The page's copy of what the server holds.
 *
 * A plain object plus a subscribe list. No business logic lives here: views read it,
 * actions write through `api` and then refresh it, so the server stays the single
 * source of truth and a failed write never leaves the screen showing something that
 * was not saved.
 */

import { api } from "./api.js";

export const state = {
  ready: false,
  owner: true,
  dogs: [],
  chats: [],
  activeDogId: null,
  chatId: null,
  search: "",
  health: null,
};

const listeners = new Set();

export const subscribe = (fn) => {
  listeners.add(fn);
  return () => listeners.delete(fn);
};

export function emit() {
  for (const fn of listeners) fn(state);
}

/* ------------------------------------------------------------------ loading */

export async function refresh() {
  const lib = await api.library();
  state.owner = lib.owner;
  state.dogs = lib.dogs || [];
  state.chats = lib.chats || [];
  state.activeDogId = lib.active_dog_id;
  state.ready = true;
  emit();
  return lib;
}

/* ------------------------------------------------------------------ reading */

export const dogById = (id) => state.dogs.find((d) => d.id === id) || null;

/* Slug matching, so /dogs/<name> works without an id in the address bar. Kept
 * identical to library.slug on the server; the id stays the real key. */
export const slug = (s) =>
  (s || "").trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") ||
  "dog";

export const dogByKey = (key) =>
  dogById(key) || state.dogs.find((d) => slug(d.name) === slug(key)) || null;

export const activeDog = () => dogById(state.activeDogId);

export const chatById = (id) => state.chats.find((c) => c.id === id) || null;

/* Search looks at the conversation, not just the title.
 *
 * The phrase you remember is usually from the middle of an answer, which is why the
 * server keeps a capped `search_text` per chat. */
export function visibleChats() {
  const q = state.search.trim().toLowerCase();
  if (!q) return state.chats;
  return state.chats.filter((c) =>
    `${c.title || ""} ${c.search_text || ""} ${c.video_name || ""}`
      .toLowerCase()
      .includes(q)
  );
}

export const chatsForDog = (dogId) => state.chats.filter((c) => c.dog_id === dogId);

/* ------------------------------------------------------------------ writing */

export async function addDog(data) {
  const dog = await api.createDog(data);
  await refresh();
  return dog;
}

/* Edits land on the server before the page claims they are saved.
 *
 * The field is also patched locally first so typing never stutters while the request
 * is in flight - but a failure refreshes from the server, so the screen can't end up
 * showing an edit that was not stored. */
export async function saveDog(id, patch) {
  const dog = dogById(id);
  if (dog) Object.assign(dog, patch);
  emit();
  try {
    await api.updateDog(id, patch);
  } finally {
    await refresh();
  }
}

export async function removeDog(id) {
  await api.deleteDog(id);
  await refresh();
}

export async function chooseDog(id) {
  state.activeDogId = id;
  emit();
  await api.setActiveDog(id);
  await refresh();
}

export async function newChat(dogId, title) {
  const chat = await api.createChat(dogId ?? state.activeDogId ?? null, title || "");
  await refresh();
  return chat;
}

export async function removeChat(id) {
  await api.deleteChat(id);
  await refresh();
}
