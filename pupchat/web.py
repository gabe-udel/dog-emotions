"""A local Gradio page: upload a video, ask, keep chatting.

    .venv\\Scripts\\python.exe -m pupchat.web

Gradio rather than Streamlit because `gr.ChatInterface` gives the conversation and the
streaming progress messages for free, and because Streamlit re-runs the whole script on
every interaction - which on a pipeline where one pass costs minutes of CPU is the wrong
execution model.

Progress matters here more than it usually would: the keypoint pass is roughly 1.2
seconds per frame on CPU, so a first question on a fresh video takes minutes. The page
says what it is doing rather than appearing hung.
"""
from __future__ import annotations

from pathlib import Path

from .config import LICENCE_NOTICE, Config, load_env

INTRO = """
# Pupchat

Upload a short clip of your dog and ask a question about it.

**First question on a new video takes a few minutes** - the keypoint model runs on CPU
at about 1.2 seconds per frame. Follow-up questions are fast: the video is measured
once, cached by its file hash, and never reprocessed.
"""


def build():
    import gradio as gr

    load_env()
    cfg = Config()
    # Last exchange is kept so the flag button can save what was actually shown,
    # rather than re-deriving it from the chat widget's rendering.
    state = {"history": None, "video": None, "last": None}

    def respond(message, chat_history, video):
        from .client import ask_pupchat

        if not video:
            chat_history = chat_history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Upload a video first."}]
            return chat_history, ""

        if video != state["video"]:          # new clip, new conversation
            state["history"] = None
            state["video"] = video

        notes: list[str] = []

        def progress(*a, **k):
            notes.append(" ".join(str(x) for x in a))

        try:
            res = ask_pupchat(video, message, history=state["history"], cfg=cfg,
                              progress=progress)
        except SystemExit as e:              # missing key, missing file
            chat_history = chat_history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"**Setup needed**\n\n```\n{e}\n```"}]
            return chat_history, ""

        state["history"] = res["history"]
        state["last"] = {"question": message, "answer": res["answer"],
                         "evidence_text": res["evidence_text"],
                         "provider": res["provider"], "model": res["model"],
                         "hash": res["hash"]}
        cost = res["cost_usd"]
        footer = (f"\n\n<sub>{res['seconds']:.0f}s"
                  + (f" · ${cost:.4f}" if cost == cost else "")
                  + "</sub>")
        chat_history = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": res["answer"] + footer}]
        return chat_history, ""

    def flag(note, video):
        from .feedback import flag_answer

        last = state["last"]
        if not last:
            return "Ask something first - there is no answer to flag yet."
        rec = flag_answer(video, last["question"], last["answer"], note or "",
                          evidence_text=last["evidence_text"],
                          provider=last["provider"], model=last["model"],
                          video_hash=last["hash"])
        if rec.get("clip"):
            return f"Saved as `{rec['id']}`. Clip copied to `{rec['clip']}`."
        return f"Saved as `{rec['id']}` (clip not copied)."

    def show_evidence(video):
        if not video:
            return "Upload a video first."
        from .pipeline import analyse
        return analyse(video, cfg, progress=lambda *a, **k: None)["evidence_text"]

    with gr.Blocks(title="Pupchat") as demo:
        gr.Markdown(INTRO)
        gr.Markdown(f"> {LICENCE_NOTICE}")
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="Your dog's video", height=320)
                ev_btn = gr.Button("Show the measurements", variant="secondary")
                ev_box = gr.Textbox(label="Evidence sent to the model", lines=16,
                                    max_lines=30, interactive=False)
            with gr.Column(scale=2):
                # Gradio 6 dropped the `type` kwarg; the messages format is the only one now.
                chat = gr.Chatbot(label="Pupchat", height=460)
                msg = gr.Textbox(label="Your question", placeholder="What is he doing?",
                                 autofocus=True)
                gr.Markdown("*Educational guidance only. For aggression, bite risk, "
                            "pain or illness, see a vet or a certified behaviourist.*")
                with gr.Accordion("Flag this answer", open=False):
                    gr.Markdown(
                        "Saves the clip, your question, the answer and the evidence "
                        "so the case can be re-run later against a changed prompt, "
                        "model or feature set.")
                    note = gr.Textbox(label="What was wrong with it?", lines=2,
                                      placeholder="e.g. called it playful; ears are "
                                                  "clearly pinned")
                    flag_btn = gr.Button("Flag", variant="stop")
                    flag_out = gr.Markdown()

        msg.submit(respond, [msg, chat, video], [chat, msg])
        ev_btn.click(show_evidence, [video], [ev_box])
        flag_btn.click(flag, [note, video], [flag_out])

    return demo


def main(port: int = 7860, open_browser: bool = True) -> None:
    # No show_api kwarg: Gradio 6 removed it, and passing it raises on launch. That
    # went unnoticed because the page was only ever checked by constructing it with
    # build(); constructing a Blocks never touches launch's signature.
    build().launch(server_name="127.0.0.1", server_port=port,
                   inbrowser=open_browser)


if __name__ == "__main__":
    main()
