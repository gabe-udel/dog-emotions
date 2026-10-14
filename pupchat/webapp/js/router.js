/* Real URLs, so Back works and a link can be shared.
 *
 *   /                 the welcome screen
 *   /chat/<id>        one conversation
 *   /dogs/<name>      a dog's profile
 *   /dogs/new         a blank profile to fill in
 *
 * The server serves index.html for all of these (see api.SpaFiles), so a refresh or a
 * pasted link lands in the right place rather than on "Not Found".
 *
 * Sending a message deliberately does NOT navigate - the same rule ChatGPT follows.
 * A question is a change to the conversation you are already in, and pushing a history
 * entry for it would make Back step through your own messages.
 */

const routes = [
  { pattern: /^\/chat\/([^/]+)\/?$/, view: "chat", key: "chatId" },
  { pattern: /^\/dogs\/new\/?$/, view: "dog", key: "dogKey", value: null },
  { pattern: /^\/dogs\/([^/]+)\/?$/, view: "dog", key: "dogKey" },
  { pattern: /^\/dogs\/?$/, view: "dog", key: "dogKey", value: null },
];

export function parse(path) {
  for (const r of routes) {
    const m = r.pattern.exec(path);
    if (!m) continue;
    const value = "value" in r ? r.value : decodeURIComponent(m[1] ?? "");
    return { view: r.view, [r.key]: value, isNew: /\/dogs\/new\/?$/.test(path) };
  }
  return { view: "welcome" };
}

let onChange = () => {};

export const onRoute = (fn) => {
  onChange = fn;
};

export const current = () => parse(location.pathname);

/* `replace` is for corrections - landing on a chat that no longer exists, or naming a
 * dog that has just been created. Those must not leave a dead entry in history for
 * Back to return to. */
export function go(path, { replace = false } = {}) {
  if (path === location.pathname) {
    onChange(current());
    return;
  }
  history[replace ? "replaceState" : "pushState"]({}, "", path);
  onChange(current());
}

export function start() {
  window.addEventListener("popstate", () => onChange(current()));

  /* One listener for every internal link, rather than one per element.
   *
   * Rows are drawn from <template>s as the library changes, so per-element handlers
   * would have to be re-attached on every render and any missed one would fall back to
   * a full page load - which looks like a flash and loses the page's state. */
  document.addEventListener("click", (e) => {
    const a = e.target.closest("a[href]");
    if (!a) return;
    const href = a.getAttribute("href");
    if (!href || href.startsWith("http") || a.target === "_blank") return;
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
    e.preventDefault();
    go(href);
  });

  onChange(current());
}
