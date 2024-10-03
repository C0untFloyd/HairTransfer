import os
import subprocess
import tempfile
import cv2
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import transforms
from torchvision.utils import save_image

from models.Net import get_segmentation

current_video_path = None
current_frame_total = 0
current_capture = None


def equal_replacer(images: list[torch.Tensor]) -> list[torch.Tensor]:
    for i in range(len(images)):
        if images[i].dtype is torch.uint8:
            images[i] = images[i] / 255

    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            if torch.allclose(images[i], images[j]):
                images[j] = images[i]
    return images


class DilateErosion:
    def __init__(self, dilate_erosion=5, device='cuda'):
        self.dilate_erosion = dilate_erosion
        self.weight = torch.Tensor([
            [False, True, False],
            [True, True, True],
            [False, True, False]
        ]).float()[None, None, ...].to(device)

    def hair_from_mask(self, mask):
        mask = torch.where(mask == 13, torch.ones_like(mask), torch.zeros_like(mask))
        mask = F.interpolate(mask, size=(256, 256), mode='nearest')
        dilate, erosion = self.mask(mask)
        return dilate, erosion

    def mask(self, mask):
        masks = mask.clone().repeat(*([2] + [1] * (len(mask.shape) - 1))).float()
        sum_w = self.weight.sum().item()
        n = len(mask)

        for _ in range(self.dilate_erosion):
            masks = F.conv2d(masks, self.weight,
                             bias=None, stride=1, padding='same', dilation=1, groups=1)
            masks[:n] = (masks[:n] > 0).float()
            masks[n:] = (masks[n:] == sum_w).float()

        hair_mask_dilate, hair_mask_erode = masks[:n], masks[n:]

        return hair_mask_dilate, hair_mask_erode


def poisson_image_blending(final_image, face_image, dilate_erosion=30, maxn=115):
    dilate_erosion = DilateErosion(dilate_erosion=dilate_erosion)
    transform = transforms.ToTensor()

    if isinstance(face_image, str):
        face_image = transform(Image.open(face_image))
    elif not isinstance(face_image, torch.Tensor):
        face_image = transform(face_image)

    final_mask = get_segmentation(final_image.cuda().unsqueeze(0), resize=False)
    face_mask = get_segmentation(face_image.cuda().unsqueeze(0), resize=False)

    hair_target = torch.where(final_mask == 13, torch.ones_like(final_mask),
                              torch.zeros_like(final_mask))
    hair_face = torch.where(face_mask == 13, torch.ones_like(face_mask),
                            torch.zeros_like(face_mask))

    final_mask = F.interpolate(((1 - hair_target) * (1 - hair_face)).float(), size=(1024, 1024), mode='bicubic')
    dilation, _ = dilate_erosion.mask(1 - final_mask)
    mask_save = 1 - dilation[0]

    with tempfile.TemporaryDirectory() as temp_dir:
        final_image_path = os.path.join(temp_dir, 'final_image.png')
        face_image_path = os.path.join(temp_dir, 'face_image.png')
        mask_path = os.path.join(temp_dir, 'mask_save.png')
        save_image(final_image, final_image_path)
        save_image(face_image, face_image_path)
        save_image(mask_save, mask_path)

        out_image_path = os.path.join(temp_dir, 'out_image_path.png')
        result = subprocess.run(
            ["fpie", "-s", face_image_path, "-m", mask_path, "-t", final_image_path, "-o", out_image_path, "-n",
             str(maxn), "-b", "taichi-gpu", "-g", "max"],
            check=True
        )

        return Image.open(out_image_path), Image.open(mask_path)


def adapt_to_ref_image(source, reference):
    source = source.astype(np.uint8)
    reference = reference.astype(np.uint8)

    # # Get RGB values at position (0,0) for both images
    # original_rgb = source[0, 0]
    # reference_rgb = reference[0, 0]

    # # Calculate the darkening factor
    # factor = np.mean(original_rgb / reference_rgb)
    # factor = 1.0 - (factor - 1.0)

    #factor = 0.97
    # Apply the darkening factor to the entire image
    #return (source * factor).astype(np.uint8)
    # Convert images from BGR to LAB color space
    original_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    # Get LAB values at position (0,0) for both images
    original_lab_value = original_lab[4, 4]
    reference_lab_value = reference_lab[4, 4]

    # Calculate the darkening factor based on the L channel
    #    factor = float(original_lab_value[0]) / float(reference_lab_value[0])
    factor = 0.969
    # Convert L channel to float
    l_channel = original_lab[:,:,0].astype(np.float32)
    # Apply the darkening factor to the L channel
    l_channel *= factor
    # Clip values to [0, 255] range and convert back to uint8
    original_lab[:,:,0] = np.clip(l_channel, 0, 255).astype(np.uint8)
    # Convert back to BGR color space
    return cv2.cvtColor(original_lab, cv2.COLOR_LAB2BGR)



def histogram_matching(source, reference):
    # Convert images to LAB color space
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)

    # Split the LAB images into channels
    source_l, source_a, source_b = cv2.split(source_lab)
    reference_l, reference_a, reference_b = cv2.split(reference_lab)

    # Ensure L channels are 8-bit unsigned integers
    source_l = source_l.astype(np.uint8)
    reference_l = reference_l.astype(np.uint8)

    # Calculate histograms for L channel
    source_hist, _ = np.histogram(source_l.flatten(), 256, [0, 256])
    reference_hist, _ = np.histogram(reference_l.flatten(), 256, [0, 256])

    # Calculate cumulative distribution functions (CDFs)
    source_cdf = source_hist.cumsum()
    reference_cdf = reference_hist.cumsum()

    # Normalize CDFs
    source_cdf_normalized = source_cdf / source_cdf[-1]
    reference_cdf_normalized = reference_cdf / reference_cdf[-1]

    # Create lookup table
    lookup_table = np.zeros(256, dtype=np.uint8)
    j = 0
    for i in range(256):
        while j < 256 and reference_cdf_normalized[j] <= source_cdf_normalized[i]:
            j += 1
        lookup_table[i] = min(j, 255)  # Ensure values don't exceed 255

    # Reshape lookup table to a column vector
    lookup_table = lookup_table.reshape(-1, 1)

# Apply lookup table to source L channel
    source_l_matched = cv2.LUT(source_l, lookup_table)

    # Ensure all channels have the same data type
    source_l_matched = source_l_matched.astype(np.uint8)
    source_a = source_a.astype(np.uint8)
    source_b = source_b.astype(np.uint8)
    # Merge channels and convert back to BGR
    result_lab = cv2.merge([source_l_matched, source_a, source_b])
    result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    return result


def fix_color(extend_vision_frame_raw, extend_vision_frame):
	color_difference = compute_color_difference(extend_vision_frame_raw, extend_vision_frame, (48, 48))
	color_difference_mask = create_static_box_mask( (extend_vision_frame.shape[0], extend_vision_frame.shape[1]), 1.0, (0, 0, 0, 0))
	color_difference_mask = np.stack((color_difference_mask, ) * 3, axis = -1)
	extend_vision_frame = normalize_color_difference(color_difference, color_difference_mask, extend_vision_frame)
	return extend_vision_frame


def compute_color_difference(extend_vision_frame_raw, extend_vision_frame, size):
	extend_vision_frame_raw = extend_vision_frame_raw.astype(np.float32) / 255
	extend_vision_frame_raw = cv2.resize(extend_vision_frame_raw, size, interpolation = cv2.INTER_AREA)
	extend_vision_frame = extend_vision_frame.astype(np.float32) / 255
	extend_vision_frame = cv2.resize(extend_vision_frame, size, interpolation = cv2.INTER_AREA)
	color_difference = extend_vision_frame_raw - extend_vision_frame
	return color_difference


def normalize_color_difference(color_difference, color_difference_mask, extend_vision_frame):
	#color_difference = cv2.resize(color_difference, extend_vision_frame.shape[:2][::-1], interpolation = cv2.INTER_CUBIC)
	color_difference = cv2.resize(color_difference, (extend_vision_frame.shape[1], extend_vision_frame.shape[0]), interpolation = cv2.INTER_CUBIC)
	color_difference_mask = 1 - color_difference_mask.clip(0, 0.75)
	extend_vision_frame = extend_vision_frame.astype(np.float32) / 255
	extend_vision_frame += color_difference * color_difference_mask
	extend_vision_frame = extend_vision_frame.clip(0, 1)
	extend_vision_frame = np.multiply(extend_vision_frame, 255).astype(np.uint8)
	return extend_vision_frame

def create_static_box_mask(crop_size, face_mask_blur, face_mask_padding):
	blur_amount = int(crop_size[0] * 0.5 * face_mask_blur)
	blur_area = max(blur_amount // 2, 1)
	box_mask = np.ones(crop_size).astype(np.float32)
	box_mask[:max(blur_area, int(crop_size[1] * face_mask_padding[0] / 100)), :] = 0
	box_mask[-max(blur_area, int(crop_size[1] * face_mask_padding[2] / 100)):, :] = 0
	box_mask[:, :max(blur_area, int(crop_size[0] * face_mask_padding[3] / 100))] = 0
	box_mask[:, -max(blur_area, int(crop_size[0] * face_mask_padding[1] / 100)):] = 0
	if blur_amount > 0:
		box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)
	return box_mask



def list_image_files(directory):
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []

    for entry in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, entry)
        if os.path.isfile(file_path):
            file_extension = Path(file_path).suffix.lower()
            if file_extension in image_extensions:
                image_files.append(entry)

    return image_files


def get_image_frame(filename: str):
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
    except:
        print(f"Exception reading {filename}")
    return None

    
def get_video_frame(video_path: str, frame_number: int = 0):
    global current_video_path, current_capture, current_frame_total

    if video_path != current_video_path:
        release_video()
        current_capture = cv2.VideoCapture(video_path)
        current_video_path = video_path
        current_frame_total = current_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    current_capture.set(cv2.CAP_PROP_POS_FRAMES, min(current_frame_total, frame_number - 1))
    has_frame, frame = current_capture.read()
    if has_frame:
        return frame
    return None

def release_video():
    global current_capture    

    if current_capture is not None:
        current_capture.release()
        current_capture = None
        

def get_video_frame_total(video_path: str) -> int:
    capture = cv2.VideoCapture(video_path)
    video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return video_frame_total
