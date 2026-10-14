/* The model's answer, as the design's cards.
 *
 * The prompt asks for five headings - What I see / What it likely means / Do / Don't /
 * Try this next - but says to use fewer, or differently named ones, when the question
 * calls for it. So this renders whatever headings come back: the five it knows get the
 * design's colours, and anything else gets a plain card rather than being dropped.
 *
 * Deliberately more permissive than the server's answer_check heading regex. Drift
 * between the two can then only make the page prettier, never make it lose text.
 */

import { tpl } from "./views.js";

/* Known sections and how the design paints them. The "Do"/"Don't" colours are the
 * mockup's own: a left border and a tinted ground. */
const SECTIONS = {
  "what i see": { icon: "visibility", tone: "neutral" },
  "what it likely means": { icon: "psychology", tone: "neutral" },
  do: { icon: "check_circle", tone: "do" },
  "don't": { icon: "close", tone: "dont" },
  dont: { icon: "close", tone: "dont" },
  "try this next": { icon: "school", tone: "steps" },
};

const TONE = {
  neutral: "bg-surface-container-low border-outline-variant/40 text-on-surface",
  do: "bg-secondary-container/40 border-l-[3px] border-secondary text-on-surface",
  dont: "bg-error-container/40 border-l-[3px] border-error text-on-surface",
  steps: "bg-surface-container border-outline-variant/40 text-on-surface",
};

/* Split on bold or markdown headings.
 *
 * Accepts **Heading**, ## Heading and "Heading:" because the model is told to bold them
 * but is not punished for choosing otherwise - and an answer whose text we refused to
 * display because of its punctuation would be the worst possible failure here.
 */
export function parseSections(text) {
  const lines = (text || "").replace(/\r/g, "").split("\n");
  const out = [];
  let current = { title: null, body: [] };

  const headingOf = (line) => {
    const s = line.trim();
    let m = /^\*\*(.+?)\*\*:?\s*$/.exec(s) || /^__(.+?)__:?\s*$/.exec(s);
    if (m) return m[1].trim();
    m = /^#{1,4}\s+(.+?)\s*:?\s*$/.exec(s);
    if (m) return m[1].trim();
    // "Heading:" alone on a line, but not a sentence that happens to contain a colon.
    m = /^([A-Z][^.?!:]{2,40}):\s*$/.exec(s);
    if (m) return m[1].trim();
    return null;
  };

  for (const line of lines) {
    const h = headingOf(line);
    if (h) {
      if (current.title || current.body.join("").trim()) out.push(current);
      current = { title: h, body: [] };
    } else {
      current.body.push(line);
    }
  }
  if (current.title || current.body.join("").trim()) out.push(current);
  return out.map((s) => ({ title: s.title, body: s.body.join("\n").trim() }));
}

/* Bullets and numbered steps become real lists; everything else stays paragraphs. */
function paintBody(container, body) {
  const blocks = body.split(/\n{2,}/);
  for (const block of blocks) {
    const lines = block.split("\n").map((l) => l.trim()).filter(Boolean);
    if (!lines.length) continue;

    const bulleted = lines.every((l) => /^[-*•]\s+/.test(l));
    const numbered = lines.every((l) => /^\d+[.)]\s+/.test(l));

    if (bulleted || numbered) {
      const list = document.createElement(numbered ? "ol" : "ul");
      list.className =
        (numbered ? "list-decimal" : "list-disc") +
        " pl-5 space-y-1 font-body-sm text-body-sm";
      for (const l of lines) {
        const li = document.createElement("li");
        li.textContent = l.replace(/^([-*•]|\d+[.)])\s+/, "");
        list.appendChild(li);
      }
      container.appendChild(list);
    } else {
      const p = document.createElement("p");
      p.className = "font-body-md text-body-md leading-relaxed";
      // textContent, never innerHTML: this is model output rendered on the owner's own
      // origin, beside their dog's details.
      p.textContent = lines.join(" ").replace(/\*\*(.+?)\*\*/g, "$1");
      container.appendChild(p);
    }
  }
}

export function renderAnswer(text, { sources = [], meta = "" } = {}) {
  const card = tpl("answer-card");
  // The <article> is the design's insight container. It still holds the mockup's own
  // sections, so empty it and keep the chrome around it.
  const shell = card.querySelector("article");
  if (!shell) return card;
  shell.textContent = "";

  const sections = parseSections(text);
  for (const section of sections) {
    const known = section.title ? SECTIONS[section.title.toLowerCase()] : null;
    const box = document.createElement("div");
    box.className =
      "rounded-xl p-4 space-y-2 border " + TONE[known?.tone || "neutral"];

    if (section.title) {
      const head = document.createElement("div");
      head.className = "flex items-center gap-2";
      const icon = document.createElement("span");
      icon.className = "material-symbols-outlined text-[18px] text-primary";
      icon.textContent = known?.icon || "lightbulb";
      const label = document.createElement("span");
      label.className =
        "font-label-sm text-label-sm uppercase tracking-wider font-semibold";
      label.textContent = section.title;
      head.append(icon, label);
      box.appendChild(head);
    }
    paintBody(box, section.body);
    shell.appendChild(box);
  }

  if (sources.length) shell.appendChild(sourceList(sources));

  if (meta) {
    const foot = document.createElement("p");
    foot.className = "font-label-sm text-label-sm text-outline pt-1";
    foot.textContent = meta;
    shell.appendChild(foot);
  }
  return card;
}

/* Where the advice was found.
 *
 * Every entry comes from the provider's own grounding metadata, never from the answer
 * text - which is what makes an invented link impossible rather than merely unlikely.
 * A real page is still not a proven claim, and the note says so.
 */
export function sourceList(sources) {
  const box = document.createElement("div");
  box.className = "rounded-xl p-4 border border-outline-variant/40 bg-surface-container-low space-y-2";

  const head = document.createElement("div");
  head.className = "flex items-center gap-2";
  const icon = document.createElement("span");
  icon.className = "material-symbols-outlined text-[18px] text-primary";
  icon.textContent = "explore";
  const label = document.createElement("span");
  label.className = "font-label-sm text-label-sm uppercase tracking-wider font-semibold";
  label.textContent = "Sources";
  head.append(icon, label);
  box.appendChild(head);

  const list = document.createElement("ol");
  list.className = "list-decimal pl-5 space-y-1 font-body-sm text-body-sm";
  for (const s of sources) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = s.uri;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.className = "text-primary hover:underline";
    a.textContent = s.title || s.domain || s.uri;
    li.appendChild(a);
    if (s.domain) {
      const dom = document.createElement("span");
      dom.className = "text-outline";
      dom.textContent = ` — ${s.domain}`;
      li.appendChild(dom);
    }
    list.appendChild(li);
  }
  box.appendChild(list);

  const note = document.createElement("p");
  note.className = "font-label-sm text-label-sm text-outline/80";
  note.textContent =
    "These are where the advice was found, not proof that it is right.";
  box.appendChild(note);
  return box;
}
