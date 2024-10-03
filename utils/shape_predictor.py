from pathlib import Path

import PIL
import PIL.ImageTransform
import insightface
import numpy as np
import scipy
import scipy.ndimage
import torch
from PIL import Image
from torchvision import transforms as T
import cv2
import globals

from skimage import transform as trans
from utils.image_utils import get_video_frame, fix_color, histogram_matching, adapt_to_ref_image

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


def get_all_faces(frame):
    try:
        faces = get_face_analyser().get(frame)
        return sorted(faces, key=lambda x: x.bbox[0])
    except:
        return None

def get_first_face(frame):
    try:
        faces = get_face_analyser().get(frame)
        return min(faces, key=lambda x: x.bbox[0])
    except:
        return None
    

def preview_swap(frame):
    global hair_fast

    face = get_first_face(frame)
    if face is None:
        return None

    face = align_crop_head(frame, face)
    shape = globals.INPUT_FACESETS[0].faces[0]
    color = shape if shape.use_color else face
    final_image = globals.hair_fast.swap(face.tensor, shape.tensor, color.tensor, align=False)
    img = T.functional.to_pil_image(final_image)
    img = np.array(img)
    merged_result = paste_back(face, img, frame)
    return cv2.cvtColor(merged_result, cv2.COLOR_RGB2BGR)






def get_landmark_from_tensors(tensors: list[torch.Tensor | Image.Image | np.ndarray]):
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

# alignment code from insightface https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py

arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (5, 2)
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    elif image_size % 128 == 0:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    elif image_size % 512 == 0:
        ratio = float(image_size) / 512.0
        diff_x = 32.0 * ratio

    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M


def align_crop(img, landmark, image_size=112, mode="arcface"):
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped, M



def test(data):
    if isinstance(data, list):
        data = [data]
    img = np.array(data)
    dets = get_face_analyser().get(img)
    if len(dets) == 0:
        raise ValueError(f"No faces detected in the image")
    elif len(dets) == 1:
        print(f"Number of faces detected: {len(dets)}")
    else:
        print(f"Number of faces detected: {len(dets)}, get largest face")
    face = min(dets, key=lambda x: x.bbox[0])
    aligned_img, M = align_crop(img, face.kps, 1024)
    cv2.imwrite("test.png", aligned_img)
    face.matrix = M


def extract_face_images(source_filename, video_info, extra_padding=-1.0):
    face_data = []
    source_image = None

    if video_info[0]:
        frame = get_video_frame(source_filename, video_info[1])
        if frame is not None:
            source_image = frame
        else:
            return face_data
    else:
        source_image = cv2.imdecode(np.fromfile(source_filename, dtype=np.uint8), cv2.IMREAD_COLOR)

    faces = get_all_faces(source_image)
    if faces is None:
        return face_data
    
    for face in faces:
        face = align_crop_head(source_image, face)
        face_data.append(face)
    return face_data


    

def align_crop_head(img, face):
    lm = np.array([[tt[0], tt[1]] for tt in face.landmark_3d_68])
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
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

    output_size = 1024
    transform_size = 4096
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.shape[1]) / shrink)), int(np.rint(float(img.shape[0]) / shrink)))
        img = cv2.resize(img, rsize, cv2.INTER_AREA)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.shape[1]),
            min(crop[3] + border, img.shape[0]))
    if crop[2] - crop[0] < img.shape[1] or crop[3] - crop[1] < img.shape[0]:
        #[startY:endY, startX:endX]
        img = img[crop[1]:crop[3], crop[0]:crop[2]]
        face.CropBB = crop
        quad -= crop[0:2]
        face.CropBB = crop
    else:
        face.CropBB = None

    # Pad.
    enable_padding = False
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.shape[1] + border, 0),
           max(pad[3] - img.shape[0] + border, 0))
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
        quad += pad[:2]

    face.PaddedShape = img.shape

    dst = np.array(
    [
        [0, 0],
        [0, transform_size-1],
        [transform_size -1 , transform_size - 1],
        [transform_size - 1, 0]
    ],
    dtype=np.float64,
    )
    tform = trans.SimilarityTransform()
    tform.estimate(quad, dst)
    M = tform.params[0:2, :]
    face.Matrix = M
    img = cv2.warpAffine(img, M, (transform_size, transform_size), borderValue=0.0)

    if output_size < transform_size:
        img = cv2.resize(img,(output_size, output_size), cv2.INTER_AREA)

    img = img.astype(np.uint8)
    transform = T.ToTensor()
    tensor = transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).clamp(0, 1)
    face.img = img
    face.tensor = tensor
    return face


def paste_back(face, fake_img, target_img):
    transform_size = (4096, 4096)
    fake_img = cv2.resize(fake_img,transform_size, cv2.INTER_CUBIC)

    if face.CropBB is None:
        crop_target = target_img
    else:
        crop_target = target_img[face.CropBB[1]:face.CropBB[3], face.CropBB[0]:face.CropBB[2]]

    fake_img = adapt_to_ref_image(fake_img, crop_target).astype(np.float32)

    scale = 1.0
    M_scale = face.Matrix * scale
    IM = cv2.invertAffineTransform(M_scale)
    #cropsize = (face.CropBB[2] - face.CropBB[0], face.CropBB[3] - face.CropBB[1])
    # Generate white square sized as a upsk_face
    img_matte = np.zeros(transform_size, dtype=np.uint8)

    w = img_matte.shape[1]
    h = img_matte.shape[0]
    img_matte[0:h,0:w] = 255

    # Transform white square back to target_img
    img_matte = cv2.warpAffine(img_matte, IM,(face.PaddedShape[1], face.PaddedShape[0]), flags=cv2.INTER_NEAREST, borderValue=0.0) 
    ##Blacken the edges of face_matte by 1 pixels (so the mask in not expanded on the image edges)
    img_matte[:1,:] = img_matte[-1:,:] = img_matte[:,:1] = img_matte[:,-1:] = 0
    img_matte = blur_area(img_matte, 1, 20)

    #Normalize images to float values and reshape
    img_matte = img_matte.astype(np.float32)/255
    face_matte = np.full((face.PaddedShape[0], face.PaddedShape[1]), 255, dtype=np.uint8)
    face_matte = face_matte.astype(np.float32)/255
    img_matte = np.minimum(face_matte, img_matte)


    img_matte = np.reshape(img_matte, [img_matte.shape[0],img_matte.shape[1],1]) 
    paste_face = cv2.warpAffine(fake_img, IM, (face.PaddedShape[1], face.PaddedShape[0]), borderMode=cv2.BORDER_REPLICATE)
    # Re-assemble image
    paste_face = cv2.cvtColor(paste_face, cv2.COLOR_RGB2BGR)
    paste_face = img_matte * paste_face
    paste_face = paste_face + (1-img_matte) * crop_target.astype(np.float32)
    if face.CropBB is None:
        return paste_face.astype(np.uint8)
    #cutout_orig = target_img[face.CropBB[1]:face.CropBB[3], face.CropBB[0]:face.CropBB[2]]
    #paste_face = histogram_matching(paste_face.astype(np.uint8), cutout_orig)
    #paste_face = fix_color(paste_face.astype(np.uint8), cutout_orig)
    #paste_face = adapt_to_ref_image(paste_face, cutout_orig)
    target_img[face.CropBB[1]:face.CropBB[3], face.CropBB[0]:face.CropBB[2]] = paste_face.astype(np.uint8)
    return target_img.astype(np.uint8)


def blur_area(img_matte, num_erosion_iterations, blur_amount):
        # Detect the affine transformed white area
        mask_h_inds, mask_w_inds = np.where(img_matte==255) 
        # Calculate the size (and diagonal size) of transformed white area width and height boundaries
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds) 
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h*mask_w))
        # Calculate the kernel size for eroding img_matte by kernel (insightface empirical guess for best size was max(mask_size//10,10))
        # k = max(mask_size//12, 8)
        k = max(mask_size//(blur_amount // 2) , blur_amount // 2)
        kernel = np.ones((k,k),np.uint8)
        img_matte = cv2.erode(img_matte,kernel,iterations = num_erosion_iterations)
        #Calculate the kernel size for blurring img_matte by blur_size (insightface empirical guess for best size was max(mask_size//20, 5))
        # k = max(mask_size//24, 4) 
        k = max(mask_size//blur_amount, blur_amount//5) 
        kernel_size = (k, k)
        blur_size = tuple(2*i+1 for i in kernel_size)
        return cv2.GaussianBlur(img_matte, blur_size, 0)


def paste_back2(face, fake_img, target_img):
    transform_size = (4096, 4096)
    fake_img = cv2.resize(fake_img,transform_size, cv2.INTER_CUBIC)

    crop_target = target_img[face.CropBB[1]:face.CropBB[3], face.CropBB[0]:face.CropBB[2]]

    scale = 1.0
    M_scale = face.Matrix * scale
    IM = cv2.invertAffineTransform(M_scale)
    cropsize = (face.CropBB[2] - face.CropBB[0], face.CropBB[3] - face.CropBB[1])
    # Generate white square sized as a upsk_face
    img_matte = np.zeros(transform_size, dtype=np.uint8)

    w = img_matte.shape[1]
    h = img_matte.shape[0]
    mask_offsets = (0,0,0,0,1,20)
    top = int(mask_offsets[0] * h)
    bottom = int(h - (mask_offsets[1] * h))
    left = int(mask_offsets[2] * w)
    right = int(w - (mask_offsets[3] * w))
    img_matte[top:bottom,left:right] = 255

    # Transform white square back to target_img
    img_matte = cv2.warpAffine(img_matte, IM,cropsize, flags=cv2.INTER_NEAREST, borderValue=0.0) 
    ##Blacken the edges of face_matte by 1 pixels (so the mask in not expanded on the image edges)
    img_matte[:1,:] = img_matte[-1:,:] = img_matte[:,:1] = img_matte[:,-1:] = 0

    #Normalize images to float values and reshape
    img_matte = img_matte.astype(np.float32)/255
    # face_matte = np.full(cropsize, 255, dtype=np.uint8)
    # face_matte = face_matte.astype(np.float32)/255
    # cv2.imwrite("i1.png", face_matte)
    # img_matte = np.minimum(face_matte, img_matte)


    img_matte = np.reshape(img_matte, [img_matte.shape[0],img_matte.shape[1],1]) 
    paste_face = cv2.warpAffine(fake_img, IM, cropsize, borderMode=cv2.BORDER_REPLICATE)
    # Re-assemble image
    paste_face = cv2.cvtColor(paste_face, cv2.COLOR_RGB2BGR)
    paste_face = img_matte * paste_face
    paste_face = paste_face + (1-img_matte) * crop_target.astype(np.float32)
    target_img[face.CropBB[1]:face.CropBB[3], face.CropBB[0]:face.CropBB[2]] = paste_face
    return target_img.astype(np.uint8)




