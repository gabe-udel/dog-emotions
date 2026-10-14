/* "Open on your phone": the tunnel address, as a QR code.
 *
 * Only ever shown on the machine running the server - the endpoints behind it are
 * local-only, because the code is a credential and the first visitor handed the link
 * could otherwise re-serve it to anyone.
 */

import { $ } from "./views.js";

let pollTimer = null;

export function wireShare() {
  const btn = $("shareBtn");
  const panel = $("sharePanel");
  if (!btn || !panel) return;

  // Shown only to the local page. A visitor has no use for it and no access to it.
  const local = ["localhost", "127.0.0.1", "[::1]", ""].includes(location.hostname);
  btn.classList.toggle("hidden", !local);

  btn.addEventListener("click", () => {
    panel.classList.remove("hidden");
    refresh();
  });
  const close = () => {
    panel.classList.add("hidden");
    clearTimeout(pollTimer);
  };
  $("sharePanelClose")?.addEventListener("click", close);
  panel.addEventListener("click", (e) => {
    if (e.target === panel) close();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !panel.classList.contains("hidden")) close();
  });
}

async function refresh() {
  const body = $("shareBody");
  if (!body) return;
  let info;
  try {
    info = await (await fetch("/api/share/url")).json();
  } catch {
    body.textContent = "Could not reach Pupchat.";
    return;
  }

  body.textContent = "";
  if (!info.url) {
    body.appendChild(para(explain(info.reason)));
    // A tunnel takes a few seconds to be issued, so keep looking while the panel is up.
    if (info.reason.includes("connecting") || info.reason.includes("not written")) {
      pollTimer = setTimeout(refresh, 2000);
    }
    return;
  }

  const img = document.createElement("img");
  img.src = `/api/share/qr.png?t=${Date.now()}`;
  img.alt = "QR code for this Pupchat";
  img.className = "w-56 h-56 mx-auto rounded-xl bg-white p-2 shadow-sm";
  body.appendChild(img);

  const link = document.createElement("p");
  link.className =
    "font-body-sm text-body-sm text-on-surface break-all select-all " +
    "bg-surface-container rounded-lg p-2.5";
  link.textContent = info.url;
  body.appendChild(link);

  const row = document.createElement("div");
  row.className = "flex items-center gap-2";
  row.appendChild(button("Copy link", async () => {
    await navigator.clipboard.writeText(info.url);
  }));
  row.appendChild(button("New code", async () => {
    await fetch("/api/share/rotate", { method: "POST" });
    refresh();
  }, true));
  body.appendChild(row);

  /* Said next to the code, not buried in a doc. Anyone who photographs this screen can
   * edit the owner's dogs until the code is rotated. */
  const warn = document.createElement("p");
  warn.className =
    "font-label-sm text-label-sm text-on-error-container bg-error-container/50 rounded-lg p-2.5";
  warn.textContent =
    "This code lets any device edit your dogs. Only show it to your own phone — " +
    "press New code if anyone else sees it.";
  body.appendChild(warn);
}

function explain(reason) {
  if (reason.includes("earlier run")) {
    return "That tunnel has stopped. Start Share Pupchat.bat again for a new address.";
  }
  if (reason.includes("connecting") || reason.includes("not written")) {
    return "Waiting for Cloudflare to issue an address — a few seconds.";
  }
  return (
    "Your phone needs this Pupchat's public address, and that only exists while " +
    "Share Pupchat.bat is running. Start that, then open this again."
  );
}

const para = (text) => {
  const p = document.createElement("p");
  p.className = "font-body-sm text-body-sm text-on-surface-variant leading-relaxed";
  p.textContent = text;
  return p;
};

function button(label, onClick, subtle = false) {
  const b = document.createElement("button");
  b.type = "button";
  b.className =
    "flex-1 px-3 py-2 rounded-lg font-label-md text-label-md transition-colors " +
    (subtle
      ? "bg-surface-container hover:bg-surface-container-high text-on-surface"
      : "bg-primary hover:bg-primary-container text-on-primary");
  b.textContent = label;
  b.addEventListener("click", onClick);
  return b;
}
