"""Minimal MathTutor demo shell.

This file does not load a MiniMind checkpoint by default. It is a placeholder
for the later WebUI stage and can be expanded after the math model is trained.
"""

from __future__ import annotations


def reply(message: str) -> str:
    return (
        "MathTutor WebUI scaffold is ready. Train or select a MiniMind checkpoint "
        "before enabling live model inference.\n\n"
        f"Your message was: {message}"
    )


def main() -> None:
    try:
        import gradio as gr  # type: ignore
    except ImportError:
        print("Gradio is not installed. Install gradio later when starting the WebUI stage.")
        return

    demo = gr.ChatInterface(fn=reply, title="MiniMind-MathTutor")
    demo.launch()


if __name__ == "__main__":
    main()
