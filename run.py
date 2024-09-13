
import gradio as gr
import torchvision.transforms as T
import hashlib

from io import BytesIO
from hair_swap import HairFast, get_parser
from utils.shape_predictor import align_face
from PIL import Image
from cachetools import LRUCache


def center_crop(img):
    width, height = img.size
    side = min(width, height)

    left = (width - side) / 2
    top = (height - side) / 2
    right = (width + side) / 2
    bottom = (height + side) / 2

    img = img.crop((left, top, right, bottom))
    return img


def get_bytes(img):
    if img is None:
        return img

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return buffered.getvalue()


def resize(name):
    def resize_inner(img, align):
        global align_cache

        if name in align:
            img_hash = hashlib.md5(get_bytes(img)).hexdigest()

            if img_hash not in align_cache:
                img = align_face(img, return_tensors=False)[0]
                align_cache[img_hash] = img
            else:
                img = align_cache[img_hash]

        elif img.size != (1024, 1024):
            img = center_crop(img)
            img = img.resize((1024, 1024), Image.Resampling.LANCZOS)

        return img

    return resize_inner


def swap_hair(face, shape, color, blending, poisson_iters, poisson_erosion):
    global hair_fast
    
    if not face and not shape and not color:
        return gr.update(visible=False), gr.update(value="Need to upload a face and at least a shape or color ❗", visible=True)
    elif not face:
        return gr.update(visible=False), gr.update(value="Need to upload a face ❗", visible=True)
    elif not shape and not color:
        return gr.update(visible=False), gr.update(value="Need to upload at least a shape or color ❗", visible=True)

#    face = face.resize((512,512))
    if shape is None:
        shape = face
    if color is None:
        color = face
    final_image, face_align, shape_align, color_align = hair_fast.swap(face, shape, color, align=True)
    img = T.functional.to_pil_image(final_image)
    return img, gr.update(visible=False)
    #return gr.update(value=output, visible=True), gr.update(visible=False)


def get_demo():
    with gr.Blocks(title="Hair Transfer") as demo:
        gr.Markdown("## Hair Transfer")
        gr.Markdown(
            '<div style="display: flex; align-items: center; gap: 10px;">'
            '<span>A fork of HairFastGAN:</span>'
            '<a href="https://arxiv.org/abs/2404.01094"><img src="https://img.shields.io/badge/arXiv-2404.01094-b31b1b.svg" height=22.5></a>'
            '<a href="https://github.com/AIRI-Institute/HairFastGAN"><img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" height=22.5></a>'
            '<a href="https://huggingface.co/AIRI-Institute/HairFastGAN"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg" height=22.5></a>'
            '<a href="https://colab.research.google.com/#fileId=https://huggingface.co/AIRI-Institute/HairFastGAN/blob/main/notebooks/HairFast_inference.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=22.5></a>'
            '</div>'
        )
        with gr.Row():
            with gr.Column():
                source = gr.Image(label="Source photo to try on the hairstyle", type="pil")
                with gr.Row():
                    shape = gr.Image(label="Shape photo with desired hairstyle (optional)", type="pil")
                    color = gr.Image(label="Color photo with desired hair color (optional)", type="pil")
                with gr.Accordion("Advanced Options", open=False):
                    blending = gr.Radio(["Article", "Alternative_v1", "Alternative_v2"], value='Article',
                                        label="Color Encoder version", info="Selects a model for hair color transfer.")
                    poisson_iters = gr.Slider(0, 2500, value=0, step=1, label="Poisson iters",
                                              info="The power of blending with the original image, helps to recover more details. Not included in the article, disabled by default.")
                    poisson_erosion = gr.Slider(1, 100, value=15, step=1, label="Poisson erosion",
                                                info="Smooths out the blending area.")
                    align = gr.CheckboxGroup(["Face", "Shape", "Color"], value=["Face", "Shape", "Color"],
                                             label="Image cropping [recommended]",
                                             info="Selects which images to crop by face")
                btn = gr.Button("Get the haircut")
            with gr.Column():
                output = gr.Image(label="Your result")
                error_message = gr.Textbox(label="⚠️ Error ⚠️", visible=False, elem_classes="error-message")

#        gr.Examples(examples=[["input/0.png", "input/1.png", "input/2.png"], ["input/6.png", "input/7.png", None],
#                              ["input/10.jpg", None, "input/11.jpg"]],
#                    inputs=[source, shape, color], outputs=output)

        source.upload(fn=resize('Face'), inputs=[source, align], outputs=source)
        shape.upload(fn=resize('Shape'), inputs=[shape, align], outputs=shape)
        color.upload(fn=resize('Color'), inputs=[color, align], outputs=color)

        btn.click(fn=swap_hair, inputs=[source, shape, color, blending, poisson_iters, poisson_erosion],
                  outputs=[output, error_message])

    return demo


if __name__ == '__main__':
    global hair_fast
    
	# Init HairFast
    hair_fast = HairFast(get_parser().parse_args([]))
    align_cache = LRUCache(maxsize=10)
    demo = get_demo()
    demo.launch(inbrowser="True")
