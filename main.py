import gradio as gr

def greet(name: str):
    return f"Hello {name}!"

demo = gr.Interface(fn=greet, inputs=gr.Textbox(lines=2, placeholder="Name"), outputs="text")

demo.launch()
