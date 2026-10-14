/* The conversation: what is on screen, and what happens when you press send.
 *
 * Sending does not navigate. A question changes the conversation you are already in, so
 * pushing a history entry for it would make Back step through your own messages - the
 * same rule ChatGPT follows.
 */

import { api, ApiError } from "./api.js";
import { renderAnswer } from "./answer.js";
import { progressBlock, paintProgress, watch } from "./jobs.js";
import * as store from "./store.js";
import { state } from "./store.js";
import { paintClip } from "./video.js";
import { $, tpl, fill, viewRoot } from "./views.js";

/* The clip waiting to be sent with the next question: {name, hash, duration}. Held
 * here and not in the store, because it belongs to the composer rather than the
 * server's record of the chat. */
let attached = null;

export const attachment = () => attached;

export function setAttachment(value) {
  attached = value;
  paintAttachment();
}

/* ------------------------------------------------------------------ the stream */

const stream = () => viewRoot("chat")?.querySelector("[data-stream]");

function userTurn(message) {
  const el = tpl("user-turn");
  const bubble = el.querySelector("p");
  if (bubble) bubble.textContent = message.text || "";

  const clip = el.querySelector("[data-clip]");
  if (message.hash && clip) {
    paintClip(clip, message.hash);
  } else {
    // No clip on this turn - drop the whole video card rather than leaving its frame.
    clip?.closest(".group, .relative")?.remove();
  }
  return el;
}

function errorTurn(text) {
  const el = tpl("error-turn");
  fill(el, { text });
  return el;
}

export async function renderChat(chatId) {
  const box = stream();
  if (!box) return;
  box.textContent = "";

  let chat;
  try {
    chat = await api.chat(chatId);
  } catch (e) {
    box.appendChild(errorTurn(e.message));
    return;
  }

  for (const m of chat.messages || []) {
    if (m.role === "user") box.appendChild(userTurn(m));
    else if (m.role === "assistant") {
      box.appendChild(renderAnswer(m.text, {
        sources: m.sources || [],
        meta: [m.model, m.seconds && `${Math.round(m.seconds)}s`,
               m.cost_usd != null && `$${Number(m.cost_usd).toFixed(4)}`]
          .filter(Boolean).join(" · "),
      }));
    } else if (m.role === "error") box.appendChild(errorTurn(m.text));
  }

  // A measurement that was running when the page was closed is still running on the
  // server. Picking the poll back up beats showing a question with no answer under it.
  if (chat.pending_job_id) resumePolling(chat.pending_job_id);
  scrollToEnd();
  return chat;
}

function scrollToEnd() {
  requestAnimationFrame(() =>
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" })
  );
}

/* ------------------------------------------------------------------ attaching */

function paintAttachment() {
  const chip = $("media-chip");
  const name = $("media-name");
  if (!chip || !name) return;
  chip.classList.toggle("hidden", !attached);
  if (!attached) return;
  name.textContent = attached.ready
    ? `${attached.name} · ${attached.duration?.toFixed?.(1) ?? "?"}s`
    : `${attached.name} · measuring…`;
}

/* Upload and measure, then keep the hash.
 *
 * The clip is measured as soon as it is chosen rather than when the question is sent,
 * so the ~90 seconds overlaps with the owner typing instead of following it. */
export async function attachFile(file, onError) {
  if (!file) return;
  const limits = state.health?.upload;
  if (limits && file.size > limits.max_bytes) {
    onError(`That file is ${(file.size / 1e6).toFixed(0)} MB; the limit is ` +
            `${(limits.max_bytes / 1e6).toFixed(0)} MB.`);
    return;
  }

  setAttachment({ name: file.name, hash: null, ready: false });
  const box = stream();
  const progress = progressBlock();
  box?.appendChild(progress);
  scrollToEnd();

  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/jobs/analyse", { method: "POST", body: fd });
    const started = await res.json();
    if (!res.ok) throw new ApiError(started.detail || "That clip was not accepted.",
                                    res.status);

    const result = await watch(started.id, (snap) => paintProgress(progress, snap));
    setAttachment({ name: file.name, hash: result.hash,
                    duration: result.duration_s, ready: true });
  } catch (e) {
    setAttachment(null);
    onError(e.message);
  } finally {
    progress.remove();
  }
}

/* ------------------------------------------------------------------ sending */

let busy = false;

export async function send(text, { onError, onChat }) {
  if (busy) return;
  const question = (text || "").trim();
  if (!question) return;

  let chatId = state.chatId;

  if (!chatId && !attached?.ready) {
    onError(attached
      ? "The clip is still being measured — give it a moment."
      : "Add a 5–10 second clip with + before the first question.");
    return;
  }

  busy = true;
  try {
    if (!chatId) {
      const chat = await store.newChat(state.activeDogId, question.slice(0, 60));
      chatId = chat.id;
      state.chatId = chatId;
      onChat?.(chatId);
    }

    const box = stream();
    box?.appendChild(userTurn({ text: question, hash: attached?.hash }));
    const progress = progressBlock();
    box?.appendChild(progress);
    scrollToEnd();

    const hash = attached?.hash || store.chatById(chatId)?.hash;
    setAttachment(null);

    try {
      const fd = new FormData();
      fd.append("question", question);
      fd.append("chat_id", chatId);
      if (hash) fd.append("hash", hash);
      const res = await fetch("/api/jobs/ask", { method: "POST", body: fd });
      const started = await res.json();
      if (!res.ok) throw new ApiError(started.detail || "That did not work.",
                                      res.status);

      const result = await watch(started.id, (snap) => paintProgress(progress, snap));
      progress.remove();
      box?.appendChild(renderAnswer(result.answer, {
        sources: result.sources || [],
        meta: [result.model, `${Math.round(result.seconds)}s`,
               result.cost_usd != null && `$${Number(result.cost_usd).toFixed(4)}`]
          .filter(Boolean).join(" · "),
      }));
      if (result.truncated) {
        box?.appendChild(errorTurn(
          "That answer was cut off before it finished. Asking again usually works."));
      }
      scrollToEnd();
      await store.refresh();
    } catch (e) {
      progress.remove();
      box?.appendChild(errorTurn(e.message));
      scrollToEnd();
    }
  } finally {
    busy = false;
  }
}

function resumePolling(jobId) {
  const box = stream();
  const progress = progressBlock();
  box?.appendChild(progress);
  watch(jobId, (snap) => paintProgress(progress, snap))
    .then(() => renderChat(state.chatId))
    .catch(() => progress.remove());
}
