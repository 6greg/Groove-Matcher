import gradio as gr

from groove_matcher import *

gr.Interface(
    fn = groove_matcher,
    inputs=[gr.Audio(type="filepath", label="Arrangement"),
            gr.Number(label="bpm", minimum=70, maximum=200, precision=0, info="BPM of the arrangement"),
            gr.Dropdown(arrangements_types, label="Arrangement type")],
    outputs=[gr.Dataframe(label="Groove Match Results"),
             gr.File(file_count="multiple", label="MIDI")],
    title="Groove Matcher",
    flagging_mode="never",
).launch(allowed_paths=[str(drums_dir)], show_error=True, share=False)
