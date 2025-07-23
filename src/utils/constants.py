import re

from collections import namedtuple

# range of processed input/output images
A_MIN = 0.0
A_MAX = 1.0

SPECS = namedtuple('Specs', 'dim a_min a_max clip_min clip_max pad_constant_image pad_constant_label extension')
IMAGE = SPECS(dim=2,
              a_min=A_MIN, a_max=A_MAX,
              clip_min=0, clip_max=255,
              pad_constant_image=0, pad_constant_label=0,
              extension='.png')
VOLUME = SPECS(dim=3,
               a_min=A_MIN, a_max=A_MAX,
               clip_min=-1000, clip_max=1000,
               pad_constant_image=-1000, pad_constant_label=0,
               extension='.nii.gz')


def extension(image_path: str):
    return re.findall(r'\.png', image_path) or re.findall(r'\.nii\.gz', image_path)


def spec_extension(image_path: str):
    ext = extension(image_path)
    assert ext, 'Image must have either .png or .nii.gz extension'
    return IMAGE if ext[0] == '.png' else VOLUME
