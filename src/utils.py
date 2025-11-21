
import os, shutil, glob

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def list_images(folder):
    return [p for p in glob.glob(os.path.join(folder, '*')) if os.path.splitext(p)[1].lower() in IMG_EXTS]

def safe_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)
