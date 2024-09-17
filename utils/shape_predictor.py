import os
from pathlib import Path

import PIL
import dlib
import insightface
import numpy as np
import scipy
import scipy.ndimage
import torch
from PIL import Image
from torchvision import transforms as T

from utils.drive import open_url

FACE_ANALYSER = None


def get_face_analyser():
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        allowed_modules = ["landmark_3d_68", "detection", "recognition"]
        FACE_ANALYSER = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root="pretrained_models/buffalo_l",
            providers=['CUDAExecutionProvider'],
            allowed_modules=allowed_modules,
        )
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


def get_landmark(filepath, predictor):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)
    filepath = Path(filepath)
    print(f"{filepath.name}: Number of faces detected: {len(dets)}")
    shapes = [predictor(img, d) for k, d in enumerate(dets)]

    lms = [np.array([[tt.x, tt.y] for tt in shape.parts()]) for shape in shapes]

    return lms


def get_landmark_from_tensors_ins(tensors: list[torch.Tensor | Image.Image | np.ndarray]):
    transform = T.ToPILImage()
    images = []
    lms = []

    for k, tensor in enumerate(tensors):
        if isinstance(tensor, torch.Tensor):
            img_pil = transform(tensor)
        else:
            img_pil = tensor
        img = np.array(img_pil)
        images.append(img_pil)
        
        dets = get_face_analyser().get(img)
        if len(dets) == 0:
            raise ValueError(f"No faces detected in the image {k}.")
        elif len(dets) == 1:
            print(f"Number of faces detected: {len(dets)}")
        else:
            print(f"Number of faces detected: {len(dets)}, get largest face")
        face = min(dets, key=lambda x: x.bbox[0])
        lm = np.array([[tt[0], tt[1]] for tt in face.landmark_3d_68])
        lms.append(lm)

    return images, lms


def get_landmark_from_tensors(tensors: list[torch.Tensor | Image.Image | np.ndarray], predictor):
    detector = dlib.get_frontal_face_detector()
    transform = T.ToPILImage()
    images = []
    lms = []

    for k, tensor in enumerate(tensors):
        if isinstance(tensor, torch.Tensor):
            img_pil = transform(tensor)
        else:
            img_pil = tensor
        img = np.array(img_pil)
        images.append(img_pil)

        dets = detector(img, 1)
        if len(dets) == 0:
            raise ValueError(f"No faces detected in the image {k}.")
        elif len(dets) == 1:
            print(f"Number of faces detected: {len(dets)}")
        else:
            print(f"Number of faces detected: {len(dets)}, get largest face")

        # Find the largest face
        dets = sorted(dets, key=lambda det: det.width() * det.height(), reverse=True)
        shape = predictor(img, dets[0])
        lm = np.array([[tt.x, tt.y] for tt in shape.parts()])
        lms.append(lm)

    return images, lms


def align_face(data, is_filepath=False, return_tensors=True):
    """
    :param data: filepath or list torch Tensors
    :return: list of PIL Images
    """

    if is_filepath:
        lms = get_landmark_from_tensors_ins(data)
    else:
        if not isinstance(data, list):
            data = [data]
        images, lms = get_landmark_from_tensors_ins(data)

    imgs = []
    for num_img, lm in enumerate(lms):
        lm_chin = lm[0: 17]  # left-right
        lm_eyebrow_left = lm[17: 22]  # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]  # top-down
        lm_nostrils = lm[31: 36]  # top-down
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        lm_mouth_inner = lm[60: 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # read image
        if is_filepath:
            img = PIL.Image.open(data)
        else:
            img = images[num_img]

        output_size = 1024
        # output_size = 256
        transform_size = 4096
        enable_padding = True

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                              1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
                            PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.LANCZOS)

        # Save aligned image.
        imgs.append(img)

    if return_tensors:
        transform = T.ToTensor()
        tensors = [transform(img).clamp(0, 1) for img in imgs]
        return tensors
    return imgs
