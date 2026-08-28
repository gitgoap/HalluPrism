import sys
import argparse
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# Intercept --strength argument
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--strength", choices=["weak", "current", "strong"], required=True)
args, remaining_argv = parser.parse_known_args()

# Replace sys.argv for the main script
sys.argv = [sys.argv[0]] + remaining_argv

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import after modifying sys.argv
from experiments import run_decomposition
import src.uncertainty.visual as visual

def setup_perturbations(strength):
    def apply_blur(image: Image.Image) -> Image.Image:
        radius = {"weak": 8, "current": 15, "strong": 22}[strength]
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def apply_crop(image: Image.Image) -> Image.Image:
        # width fraction to retain: weak=0.75, current=0.5, strong=0.25
        frac = {"weak": 0.75, "current": 0.5, "strong": 0.25}[strength]
        w, h = image.size
        left = int(w * (1 - frac) / 2)
        top = int(h * (1 - frac) / 2)
        right = w - left
        bottom = h - top
        cropped = image.crop((left, top, right, bottom))
        return cropped.resize((w, h), Image.BILINEAR)

    def apply_brightness(image: Image.Image) -> Image.Image:
        factor = {"weak": 0.5, "current": 0.3, "strong": 0.15}[strength]
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def apply_noise(image: Image.Image) -> Image.Image:
        stddev = {"weak": 10, "current": 25, "strong": 50}[strength]
        arr = np.array(image).astype(np.float32)
        noise = np.random.normal(0, stddev, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    visual.PERTURBATIONS = {
        "blur_heavy": apply_blur,
        "crop_aggressive": apply_crop,
        "brightness_low": apply_brightness,
        "noise": apply_noise,
    }
    
    # Update the internal definitions in case they are used directly
    visual.apply_blur_heavy = apply_blur
    visual.apply_crop_aggressive = apply_crop
    visual.apply_brightness_low = apply_brightness
    visual.apply_noise = apply_noise

if __name__ == "__main__":
    setup_perturbations(args.strength)
    print(f"Running decomposition with {args.strength} visual perturbations.")
    run_decomposition.main()
