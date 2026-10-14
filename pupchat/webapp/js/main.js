/* Pupchat - entry point.
 *
 * The page is the Google Stitch design, merged into one document: one shell, three
 * screen bodies tagged `data-view`, and a set of <template>s cut from the mockup's own
 * rows. Nothing here draws anything the design did not already contain.
 *
 * This file currently does the four things that make the page honest to look at:
 * which screen is showing, the sidebar drawer on a phone, the error banner, and the
 * credits. Everything else - the library, jobs, answers, video - arrives in its own
 * module as the steps land.
 *
 * No API key ever reaches the browser. The page talks only to this server, which holds
 * the key in .env and never sends it on.
 */

const API = ""; // same origin

export const $ = (id) => document.getElementById(id);
export const view = (name) => document.querySelector(`[data-view="${name}"]`);

/* ------------------------------------------------------------------ views */

/* The Tailwind `hidden` CLASS, never the HTML attribute: two of the three view roots
 * carry display utilities, and `display:flex` beats the user-agent's `[hidden]`
 * rule - so the attribute would leave every view visible at once. */
export function showView(name) {
  for (const el of document.querySelectorAll("[data-view]")) {
    el.classList.toggle("hidden", el.dataset.view !== name);
  }
  closeDrawer();
}

/* Which screen a path means. The real router replaces this; the shape is already
 * right, so deep links work from the first step rather than being retrofitted. */
export function viewForPath(path) {
  if (path.startsWith("/chat/")) return "chat";
  if (path.startsWith("/dogs")) return "dog";
  return "welcome";
}

/* ------------------------------------------------------------------ sidebar */

function openDrawer() {
  document.querySelector("[data-sidebar]")?.classList.remove("-translate-x-full");
  document.querySelector("[data-scrim]")?.classList.remove("hidden");
}

export function closeDrawer() {
  const side = document.querySelector("[data-sidebar]");
  // Only on small screens: above `lg` the sidebar is a fixed column and
  // `lg:translate-x-0` already wins, but leaving the scrim up would cover the page.
  side?.classList.add("-translate-x-full");
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
  toastError(`Something went wrong: ${e.reason?.message || e.reason}`));

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

/* ------------------------------------------------------------------ boot */

async function boot() {
  showView(viewForPath(location.pathname));

  document.querySelector("[data-menu]")?.addEventListener("click", openDrawer);
  document.querySelector("[data-scrim]")?.addEventListener("click", closeDrawer);
  $("toastClose")?.addEventListener("click", () => $("toast").classList.add("hidden"));
  window.addEventListener("popstate", () => showView(viewForPath(location.pathname)));

  try {
    const h = await (await fetch(`${API}/api/health`)).json();
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
}

/* A module script is deferred, so DOMContentLoaded has usually already fired by the
 * time this runs. Waiting for it would mean waiting forever. */
boot();
