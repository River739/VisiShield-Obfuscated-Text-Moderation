import gradio as gr

def gradio_moderation_interface(comment_text):
    if not comment_text.strip():
        return "⚠️ Please enter a comment to analyze.", ""

    # Capture the pipeline execution logs
    import io
    import sys

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # Run your production pipeline
    moderate_comment(comment_text)

    sys.stdout = old_stdout
    logs = new_stdout.getvalue()

    # Determine UI badge color/status
    if "REJECTED" in logs:
        status_md = "### ❌ Status: **REJECTED (Toxic/Bypassed Content Detected)**"
    else:
        status_md = "### ✅ Status: **APPROVED (Clean Content)**"

    return status_md, logs

# Build the Gradio Web Application
demo = gr.Blocks(theme=gr.themes.Soft())

with demo:
    gr.Markdown("# 🛡️ VisiShield: Multi-Stage Visual Content Moderation Engine")
    gr.Markdown("Test how the hybrid text/computer vision pipeline catches leetspeak, spaces, and obfuscated text while bypassing false positives.")

    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(
                label="Enter Comment to Moderate",
                placeholder="Type something tricky like 'F u- C..k' or '@sshole'...",
                lines=3
            )
            submit_btn = gr.Button("Analyze Comment", variant="primary")

        with gr.Column():
            status_output = gr.Markdown()
            logs_output = gr.Code(label="Pipeline Execution Audit Logs", language="markdown")

    submit_btn.click(fn=gradio_moderation_interface, inputs=user_input, outputs=[status_output, logs_output])

    gr.Examples([
        ["This laptop is absolute sh1t, but the screen is awesome!"],
        ["F u- C..k"],
        ["bItcH is fucking @sshole."],
        ["I saw a duck swimming near a ship and it was awesome."]
    ], inputs=user_input)

# Launch the app with a shareable public link!
demo.launch(debug=True, share=True)