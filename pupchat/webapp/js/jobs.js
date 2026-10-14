/* Watching long work, and saying honestly how it is going.
 *
 * Measuring a clip is about 90 seconds. Sending the request and waiting looks exactly
 * like the server having died, so people reload - which starts a second measurement
 * behind the first.
 *
 * Polling, not Server-Sent Events: Cloudflare quick tunnels do not support SSE, and the
 * app is meant to be shared through one. A poll every 1.5s over a 90-second wait is
 * about 60 requests, which is nothing.
 */

import { ApiError } from "./api.js";
import { tpl } from "./views.js";

const POLL_MS = 1500;

/* The three the owner was promised, in order. The server's finer-grained steps
 * ("queued", "done") fold into these rather than appearing as their own rows. */
const SHOWN = [
  { key: "finding", label: "Finding your dog in the video" },
  { key: "measuring", label: "Measuring how they move" },
  { key: "asking", label: "Asking Pupchat's AI" },
];

const rank = (step) => {
  const i = SHOWN.findIndex((s) => s.key === step);
  if (step === "done") return SHOWN.length;
  return i < 0 ? -1 : i;
};

export function progressBlock() {
  const el = tpl("progress");
  const list = el.querySelector("[data-steps]");
  for (const { label } of SHOWN) {
    const row = tpl("progress-step");
    row.querySelector("[data-label]").textContent = label;
    list.appendChild(row);
  }
  return el;
}

export function paintProgress(el, snap) {
  const here = rank(snap.step);
  const rows = el.querySelectorAll("[data-steps] li");

  rows.forEach((row, i) => {
    const icon = row.querySelector("[data-icon]");
    const detail = row.querySelector("[data-detail]");
    const done = i < here;
    const active = i === here;

    icon.textContent = done ? "check_circle" : active ? "radio_button_checked" : "radio_button_checked";
    icon.className =
      "material-symbols-outlined text-[18px] " +
      (done ? "text-primary" : active ? "text-primary animate-pulse" : "text-outline-variant");
    row.className =
      "flex items-center gap-2.5 font-body-sm text-body-sm " +
      (done || active ? "text-on-surface" : "text-outline");
    detail.textContent = active ? snap.detail || "" : "";
  });

  /* A bar only where a real fraction exists, which is during the face loop. The
   * body-pose stage is one blocking call that reports nothing until it ends, so a bar
   * there would be interpolated against the clock - a measurement nobody made. */
  const wrap = el.querySelector("[data-bar-wrap]");
  const bar = el.querySelector("[data-bar]");
  const hasFraction = typeof snap.fraction === "number";
  wrap.classList.toggle("hidden", !hasFraction);
  if (hasFraction) bar.style.width = `${Math.round(snap.fraction * 100)}%`;

  el.querySelector("[data-eta]").textContent = etaText(snap);
}

function etaText(snap) {
  if (snap.step === "failed") return "";
  if (snap.over_estimate) {
    return "This is taking longer than usual — still working.";
  }
  const left = snap.remaining_s;
  if (left == null) return "";
  if (left <= 5) return "Almost done.";
  const rounded = left > 60 ? Math.round(left / 10) * 10 : Math.round(left / 5) * 5;
  const mins = Math.floor(rounded / 60);
  const secs = rounded % 60;
  const time = mins ? `${mins} min${secs ? ` ${secs} s` : ""}` : `${secs} s`;
  return `about ${time} left — the first clip is the slow one.`;
}

/* Poll until the job ends.
 *
 * `onTick` runs on every snapshot. A 404 means the job was cleared (an hour old, or the
 * server restarted) and is reported as such, because the page may be resuming a poll
 * after a reload rather than watching one it started. */
export async function watch(jobId, onTick) {
  for (;;) {
    let snap;
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      if (res.status === 404) {
        throw new ApiError(
          "That measurement is no longer being tracked. Reload to see where it got to.",
          404
        );
      }
      snap = await res.json();
    } catch (e) {
      if (e instanceof ApiError) throw e;
      throw new ApiError("Lost contact with Pupchat while it was working.", 0);
    }

    onTick(snap);

    if (snap.step === "done") return snap.result;
    if (snap.step === "failed") throw new ApiError(snap.error || "That did not work.", 0);
    await new Promise((r) => setTimeout(r, POLL_MS));
  }
}
