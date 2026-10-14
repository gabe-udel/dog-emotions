/* The clip in the chat: a small looping thumbnail, and a popup to watch it properly.
 *
 * The bytes never enter the browser's storage. A few seconds of H.264 is megabytes
 * against a localStorage quota of about 5 MB, so the page keeps only the hash the server
 * returned and streams from /api/video/<hash>, which supports Range requests - that is
 * what lets the popup seek.
 */

import { api } from "./api.js";
import { $ } from "./views.js";

/* Thumbnails play only while they are on screen.
 *
 * Every autoplaying <video> is a decoder. Half a dozen in a long conversation is real
 * battery and real jank on a phone, which is the device most likely to be on the
 * tunnel. This is not an optimisation; it is the difference between charming and hot. */
const seen = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      const v = e.target;
      if (e.isIntersecting) v.play().catch(() => {});
      else v.pause();
    }
  },
  { rootMargin: "120px" }
);

/* `muted` and `playsinline` are both required, not stylistic: without muted no browser
 * will autoplay, and without playsinline iOS Safari takes the video fullscreen instead
 * of playing it in the bubble. */
export function paintClip(el, hash) {
  if (!el || !hash) return;
  el.src = api.videoUrl(hash);
  el.muted = true;
  el.loop = true;
  el.playsInline = true;
  el.preload = "metadata";
  el.setAttribute("muted", "");
  el.setAttribute("playsinline", "");
  el.style.cursor = "zoom-in";
  el.addEventListener("click", (e) => {
    e.preventDefault();
    openVideo(hash);
  });
  seen.observe(el);
}

export function openVideo(hash) {
  const modal = $("videoModal");
  const player = $("videoModalPlayer");
  if (!modal || !player) return;
  player.src = api.videoUrl(hash);
  modal.classList.remove("hidden");
  player.play().catch(() => {});
}

export function closeVideo() {
  const modal = $("videoModal");
  const player = $("videoModalPlayer");
  if (!modal || !player) return;
  modal.classList.add("hidden");
  // Pause AND drop the source: a hidden <video> left playing keeps decoding, and keeps
  // the sound going, which is baffling when nothing is on screen.
  player.pause();
  player.removeAttribute("src");
  player.load();
}

export function wireVideoModal() {
  $("videoModalClose")?.addEventListener("click", closeVideo);
  $("videoModal")?.addEventListener("click", (e) => {
    if (e.target === $("videoModal")) closeVideo();     // backdrop, not the player
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeVideo();
  });
}
