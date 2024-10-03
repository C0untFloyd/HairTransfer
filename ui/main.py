import gradio as gr
import torchvision.transforms as T
import hashlib
import utils.drive as util
import os
import globals

from utils.ProcessEntry import ProcessEntry
from utils.FaceSet import FaceSet
from utils.image_utils import get_image_frame, get_video_frame, get_video_frame_total
from io import BytesIO
from utils.shape_predictor import extract_face_images, preview_swap
from PIL import Image

IS_INPUT = True
SELECTED_FACE_INDEX = 0

SELECTED_INPUT_FACE_INDEX = 0
SELECTED_TARGET_FACE_INDEX = 0

input_faces = None
target_faces = None
face_selection = None
previewimage = None

selected_preview_index = 0

is_processing = False            
list_files_process : list[ProcessEntry] = []

current_video_fps = 50



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


def render_ui():
    with gr.Blocks(title="Hair Transfer") as gradio_ui:
        gr.Markdown("## Hair Transfer")
        with gr.Row(variant='panel'):
            with gr.Column(scale=2):
                with gr.Row():
                    input_faces = gr.Gallery(label="Input faces gallery", allow_preview=False, preview=False, height=138, columns=64, object_fit="scale-down", interactive=False)
                    target_faces = gr.Gallery(label="Target faces gallery", allow_preview=False, preview=False, height=138, columns=64, object_fit="scale-down", interactive=False)
                with gr.Row():
                    bt_move_left_input = gr.Button("⬅ Move left", size='sm')
                    bt_move_right_input = gr.Button("➡ Move right", size='sm')
                    bt_move_left_target = gr.Button("⬅ Move left", size='sm')
                    bt_move_right_target = gr.Button("➡ Move right", size='sm')
                with gr.Row():
                    bt_remove_selected_input_face = gr.Button("❌ Remove selected", size='sm')
                    bt_clear_input_faces = gr.Button("💥 Clear all", variant='stop', size='sm')
                    bt_remove_selected_target_face = gr.Button("❌ Remove selected", size='sm')
                    bt_add_local = gr.Button('Add local files from', size='sm')

                with gr.Row(variant='panel'):
                    bt_srcfiles = gr.Files(label='Source Images or Facesets', file_count="multiple", file_types=["image", ".fsz"], elem_id='filelist', height=233)
                    bt_destfiles = gr.Files(label='Target File(s)', file_count="multiple", file_types=["image", "video"], elem_id='filelist', height=233)
                with gr.Row(variant='panel'):
                    color_source = gr.Checkbox(label="Use hair color of input", value=True)
                    forced_fps = gr.Slider(minimum=0, maximum=120, value=0, label="Video FPS", info='Overrides detected fps if not 0', step=1.0, interactive=True, container=True)

            with gr.Column(scale=2):
                previewimage = gr.Image(label="Preview Image", height=576, interactive=False, visible=True, format="png")
                with gr.Row(variant='panel'):
                    fake_preview = gr.Checkbox(label="Face swap frames", value=False)
                    bt_refresh_preview = gr.Button("🔄 Refresh", variant='secondary', size='sm')
                    bt_use_face_from_preview = gr.Button("Use Face from this Frame", variant='primary', size='sm')
                with gr.Row():
                    preview_frame_num = gr.Slider(1, 1, value=1, label="Frame Number", info='0:00:00', step=1.0, interactive=True)
                with gr.Row():
                    text_frame_clip = gr.Markdown('Processing frame range [0 - 0]')
                    set_frame_start = gr.Button("⬅ Set as Start", size='sm')
                    set_frame_end = gr.Button("➡ Set as End", size='sm')
        with gr.Row(visible=False) as dynamic_face_selection:
            with gr.Column(scale=2):
                face_selection = gr.Gallery(label="Detected faces", allow_preview=False, preview=False, height=138, object_fit="cover", columns=32)
            with gr.Column():
                bt_faceselect = gr.Button("☑ Use selected face", size='sm')
                bt_cancelfaceselect = gr.Button("Done", size='sm')
            with gr.Column():
                gr.Markdown(' ') 

        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                selected_face_detection = gr.Dropdown(["First found", "All female", "All male", "All faces", "Selected face"], value="First found", label="Specify face selection for swapping")
            with gr.Column(scale=1):
                num_swap_steps = gr.Slider(1, 5, value=1, step=1.0, label="Number of swapping steps", info="More steps may increase likeness")

        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                max_face_distance = gr.Slider(0.01, 1.0, value=0.65, label="Max Face Similarity Threshold", info="0.0 = identical 1.0 = no similarity")

        with gr.Row(variant='panel'):
            with gr.Column():
                bt_start = gr.Button("▶ Start", variant='primary')
                #gr.Button("👀 Open Output Folder", size='sm').click(fn=lambda: util.open_folder(roop.globals.output_path))
            with gr.Column():
                bt_stop = gr.Button("⏹ Stop", variant='secondary', interactive=False)
            with gr.Column(scale=2):
                gr.Markdown(' ') 
        with gr.Row(variant='panel'):
            with gr.Column():
                resultfiles = gr.Files(label='Processed File(s)', interactive=False)
            with gr.Column():
                resultimage = gr.Image(type='filepath', label='Final Image', interactive=False )
                resultvideo = gr.Video(label='Final Video', interactive=False, visible=False)


        bt_srcfiles.change(fn=on_srcfile_changed, show_progress='full', inputs=bt_srcfiles, outputs=[dynamic_face_selection, face_selection, input_faces, bt_srcfiles])
        bt_clear_input_faces.click(fn=on_clear_input_faces, outputs=[input_faces])
        bt_remove_selected_input_face.click(fn=remove_selected_input_face, outputs=[input_faces])

        previewinputs = [preview_frame_num, bt_destfiles, fake_preview, selected_face_detection, max_face_distance]
        previewoutputs = [previewimage, preview_frame_num] 
        input_faces.select(on_select_input_face, None, None).success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs)



        bt_destfiles.change(fn=on_destfiles_changed, inputs=[bt_destfiles], outputs=[preview_frame_num, text_frame_clip], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='full')
        bt_destfiles.select(fn=on_destfiles_selected, outputs=[preview_frame_num, text_frame_clip, forced_fps], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
        bt_destfiles.clear(fn=on_clear_destfiles, outputs=[target_faces, selected_face_detection])
        bt_remove_selected_target_face.click(fn=remove_selected_target_face, outputs=[target_faces])

        fake_preview.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs)
        preview_frame_num.release(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='full', )
        bt_refresh_preview.click(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs)

        color_source.change(fn=on_color_input_changed, inputs=[color_source])            

        #btn.click(fn=swap_hair, inputs=[source, shape, color, blending, poisson_iters, poisson_erosion],
                  #outputs=[output, error_message])


    return gradio_ui


def on_srcfile_changed(srcfiles, progress=gr.Progress()):
    global SELECTION_FACES_DATA, IS_INPUT, input_faces, face_selection, last_image
    
    IS_INPUT = True

    if srcfiles is None or len(srcfiles) < 1:
        return gr.Column(visible=False), None, globals.ui_input_thumbs, None

    for f in srcfiles:    
        source_path = f.name
        if source_path.lower().endswith('fsz'):
            progress(0, desc="Retrieving faces from Faceset File")      
            unzipfolder = os.path.join(os.environ["TEMP"], 'faceset')
            if os.path.isdir(unzipfolder):
                files = os.listdir(unzipfolder)
                for file in files:
                    os.remove(os.path.join(unzipfolder, file))
            else:
                os.makedirs(unzipfolder)
            util.mkdir_with_umask(unzipfolder)
            util.unzip(source_path, unzipfolder)
            is_first = True
            # face_set = FaceSet()
            # for file in os.listdir(unzipfolder):
            #     if file.endswith(".png"):
            #         filename = os.path.join(unzipfolder,file)
            #         progress(0, desc="Extracting faceset")      
            #         SELECTION_FACES_DATA = extract_face_images(filename,  (False, 0))
            #         for f in SELECTION_FACES_DATA:
            #             face = f[0]
            #             face.mask_offsets = (0,0,0,0,1,20)
            #             face_set.faces.append(face)
            #             if is_first: 
            #                 image = util.convert_to_gradio(f[1])
            #                 globals.ui_input_thumbs.append(image)
            #                 is_first = False
            #             face_set.ref_images.append(get_image_frame(filename))
            # if len(face_set.faces) > 0:
            #     if len(face_set.faces) > 1:
            #         face_set.AverageEmbeddings()
            #     globals.INPUT_FACESETS.append(face_set)
                                        
        elif util.has_image_extension(source_path):
            progress(0, desc="Retrieving faces from image")      
            globals.source_path = source_path
            SELECTION_FACES_DATA = extract_face_images(globals.source_path,  (False, 0))
            progress(0.5, desc="Retrieving faces from image")
            for face in SELECTION_FACES_DATA:
                face_set = FaceSet()
                face.use_color = True
                face_set.faces.append(face)
                image = util.convert_to_gradio(face.img)
                globals.ui_input_thumbs.append(image)
                globals.INPUT_FACESETS.append(face_set)
                
    progress(1.0)
    return gr.Column(visible=False), None, globals.ui_input_thumbs,None


def on_select_input_face(evt: gr.SelectData):
    global SELECTED_INPUT_FACE_INDEX

    SELECTED_INPUT_FACE_INDEX = evt.index

def remove_selected_input_face():
    global SELECTED_INPUT_FACE_INDEX

    if len(globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        f = globals.INPUT_FACESETS.pop(SELECTED_INPUT_FACE_INDEX)
        del f
    if len(globals.ui_input_thumbs) > SELECTED_INPUT_FACE_INDEX:
        f = globals.ui_input_thumbs.pop(SELECTED_INPUT_FACE_INDEX)
        del f

    return globals.ui_input_thumbs

def on_clear_input_faces():
    globals.ui_input_thumbs.clear()
    globals.INPUT_FACESETS.clear()
    return globals.ui_input_thumbs


def on_color_input_changed(color_source):
    if len(globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        f = globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX]
        f.faces[0].use_color = color_source


def on_destfiles_changed(destfiles):
    global selected_preview_index, list_files_process, current_video_fps

    if destfiles is None or len(destfiles) < 1:
        list_files_process.clear()
        return gr.Slider(value=1, maximum=1, info='0:00:00'), ''
    
    for f in destfiles:
        list_files_process.append(ProcessEntry(f.name, 0,0, 0))

    selected_preview_index = 0
    idx = selected_preview_index    
    
    filename = list_files_process[idx].filename
    
    if util.is_video(filename) or filename.lower().endswith('gif'):
        total_frames = get_video_frame_total(filename)
        if total_frames is None or total_frames < 1:
            total_frames = 1
            gr.Warning(f"Corrupted video {filename}, can't detect number of frames!")
        else:
            current_video_fps = util.detect_fps(filename)
    else:
        total_frames = 1
    list_files_process[idx].endframe = total_frames
    if total_frames > 1:
        return gr.Slider(value=1, maximum=total_frames, info='0:00:00'), gen_processing_text(list_files_process[idx].startframe,list_files_process[idx].endframe)
    return gr.Slider(value=1, maximum=total_frames, info='0:00:00'), ''

def gen_processing_text(start, end):
    return f'Processing frame range [{start} - {end}]'


def on_destfiles_selected(evt: gr.SelectData):
    global selected_preview_index, list_files_process, current_video_fps

    if evt is not None:
        selected_preview_index = evt.index
    idx = selected_preview_index    
    filename = list_files_process[idx].filename
    fps = list_files_process[idx].fps
    if util.is_video(filename) or filename.lower().endswith('gif'):
        total_frames = get_video_frame_total(filename)
        current_video_fps = util.detect_fps(filename)
        if list_files_process[idx].endframe == 0:
            list_files_process[idx].endframe = total_frames 
    else:
        total_frames = 1
    
    if total_frames > 1:
        return gr.Slider(value=list_files_process[idx].startframe, maximum=total_frames, info='0:00:00'), gen_processing_text(list_files_process[idx].startframe,list_files_process[idx].endframe), fps
    return gr.Slider(value=1, maximum=total_frames, info='0:00:00'), gen_processing_text(0,0), fps


def remove_selected_target_face():
    if len(globals.ui_target_thumbs) > SELECTED_TARGET_FACE_INDEX:
        f = globals.TARGET_FACES.pop(SELECTED_TARGET_FACE_INDEX)
        del f
    if len(globals.ui_target_thumbs) > SELECTED_TARGET_FACE_INDEX:
        f = globals.ui_target_thumbs.pop(SELECTED_TARGET_FACE_INDEX)
        del f
    return globals.ui_target_thumbs

def on_clear_destfiles():
    globals.TARGET_FACES.clear()
    globals.ui_target_thumbs.clear()
    return globals.ui_target_thumbs, gr.Dropdown(value="First found")    



def on_preview_frame_changed(frame_num, files, fake_preview, detection_mode, face_distance):
    global SELECTED_INPUT_FACE_INDEX, current_video_fps

    timeinfo = '0:00:00'
    if files is None or selected_preview_index >= len(files) or frame_num is None:
        return None, gr.Slider(info=timeinfo)

    filename = files[selected_preview_index].name
    if util.is_video(filename) or filename.lower().endswith('gif'):
        current_frame = get_video_frame(filename, frame_num)
        if current_video_fps == 0:
            current_video_fps = 1
        secs = (frame_num - 1) / current_video_fps
        minutes = secs / 60
        secs = secs % 60
        hours = minutes / 60
        minutes = minutes % 60
        milliseconds = (secs - int(secs)) * 1000
        timeinfo = f"{int(hours):0>2}:{int(minutes):0>2}:{int(secs):0>2}.{int(milliseconds):0>3}"  
    else:
        current_frame = get_image_frame(filename)
    if current_frame is None:
        return None, gr.Slider(info=timeinfo)
    

    if not fake_preview or len(globals.INPUT_FACESETS) < 1:
        return gr.Image(value=util.convert_to_gradio(current_frame), visible=True), gr.Slider(info=timeinfo)

    face_index = SELECTED_INPUT_FACE_INDEX
    if len(globals.INPUT_FACESETS) <= face_index:
        face_index = 0
    
    current_frame = preview_swap(current_frame)

    if current_frame is None:
        return gr.Image(visible=True), gr.Slider(info=timeinfo)
    return gr.Image(value=current_frame, visible=True), gr.Slider(info=timeinfo)
