import torch
import numpy as np

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management

import latent_preview

MAX_RESOLUTION = 8192

"""
Flips an image horizontally or vertically.
"""
class mbImageFlip:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "flip_type": (["horizontal", "vertical", "both"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "flip_image"

    CATEGORY = "Mockba"

    def flip_image(self, image, flip_type):
        # Convert the input image tensor to a NumPy array
        image_np = 255.0 * image.cpu().numpy().squeeze()

        if flip_type == "horizontal":
            flipped_image_np = np.flip(image_np, axis=1)
        elif flip_type == "vertical":
            flipped_image_np = np.flip(image_np, axis=0)
        else:
            flipped_image_np = np.flip(np.flip(image_np, axis=1), axis=0)

        # Convert the flipped NumPy array back to a tensor
        flipped_image_np = flipped_image_np.astype(np.float32) / 255.0
        flipped_image_tensor = torch.from_numpy(flipped_image_np).unsqueeze(0)

        return (flipped_image_tensor,)


"""
Rotates an image by 90, 180 or 270 degrees ccw.
"""
class mbImageRot:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "rot_type": (["0", "90", "180", "270"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "rotate_image"

    CATEGORY = "Mockba"

    def rotate_image(self, image, rot_type):
        if rot_type == "0":
            return (image,)

        # Convert the input image tensor to a NumPy array
        image_np = 255.0 * image.cpu().numpy().squeeze()

        if rot_type == "90":
            rotated_image_np = np.rot90(image_np)
        elif rot_type == "180":
            rotated_image_np = np.rot90(image_np, 2)
        elif rot_type == "270":
            rotated_image_np = np.rot90(image_np, 3)
        else:
            print(
                f"Invalid rot_type. Must be either '90', '180' or '270'. No changes applied."
            )
            return (image,)

        # Convert the rotated NumPy array back to a tensor
        rotated_image_np = rotated_image_np.astype(np.float32) / 255.0
        rotated_image_tensor = torch.from_numpy(rotated_image_np).unsqueeze(0)

        return (rotated_image_tensor,)


"""
Subtracts two images. Used to measure the difference between two images.
"""
class mbImageSubtract:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_0": ("IMAGE",),
                "image_1": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "subtract"

    CATEGORY = "Mockba"

    def subtract(self, image_0, image_1):
        return (abs(image_0 - image_1),)


"""
Selects one of two latent vectors based on the value of a slider.
"""
class mbLatentSelector:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_0": ("LATENT",),
                "latent_1": ("LATENT",),
                "selected": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,  # Minimum value
                        "max": 1,  # Maximum value
                        "step": 1,  # Slider's step
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"

    CATEGORY = "Mockba"

    def run(self, latent_0, latent_1, selected: int):
        if selected == 0:
            return (latent_0,)
        else:
            return (latent_1,)


"""
Creates an empty latent image using the cpu or gpu.
"""
class mbEmptyLatentImage:

    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "Mockba"

    def generate(self, width, height, batch_size=1, device="cpu"):
        self.device = device
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=device)
        return ({"samples": latent},)


"""
creates random noise given a latent image and a seed.
optional arg skip can be used to skip and discard x number of noise generations for a given seed.
uses cpu or gpu depending on the device of the latent image.
"""
def my_prepare_noise(latent_image, seed, noise_inds=None):
    generator = torch.Generator(device=latent_image.device)
    generator.manual_seed(seed)
    if noise_inds is None:
        noise = torch.randn(
            latent_image.size(),
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            generator=generator,
            device=latent_image.device,
        )
        return noise

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1] + 1):
        noise = torch.randn(
            [1] + list(latent_image.size())[1:],
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            generator=generator,
            device=latent_image.device,
        )
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises


"""
Runs a model with a given latent image using cpu or gpu and returns the resulting latent image.
"""
def my_common_ksampler(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent,
    denoise=1.0,
    disable_noise=False,
    start_step=None,
    last_step=None,
    force_full_denoise=False,
):
    device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(
            latent_image.size(),
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            device=latent.device,
        )
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = my_prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = latent_preview.get_previewer(device, model.model.latent_format)

    pbar = comfy.utils.ProgressBar(steps)

    def callback(step, x0, x, total_steps):
        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=denoise,
        disable_noise=disable_noise,
        start_step=start_step,
        last_step=last_step,
        force_full_denoise=force_full_denoise,
        noise_mask=noise_mask,
        callback=callback,
        seed=seed,
    )
    out = latent.copy()
    out["samples"] = samples
    return (out,)


"""
Runs a model with a given latent image using cpu or gpu and returns the resulting latent image.
"""
class mbKSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "Mockba"

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
    ):
        return my_common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
        )


"""
Runs a model with a given latent image using cpu or gpu and returns the resulting latent image.
"""
class KSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"],),
                "noise_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "Mockba"

    def sample(
        self,
        model,
        add_noise,
        noise_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        denoise=1.0,
    ):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        return my_common_ksampler(
            model,
            noise_seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
            disable_noise=disable_noise,
            start_step=start_at_step,
            last_step=end_at_step,
            force_full_denoise=force_full_denoise,
        )


NODE_CLASS_MAPPINGS = {
    "mb Latent Selector": mbLatentSelector,
    "mb Image Flip": mbImageFlip,
    "mb Image Rotate": mbImageRot,
    "mb Image Subtract": mbImageSubtract,
    "mb Empty Latent Image": mbEmptyLatentImage,
    "mb KSampler": mbKSampler,
    "mb KSampler Advanced": KSamplerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mb Latent Selector": "Latent Selector",
    "mb Image Flip": "Image Flip",
    "mb Image Rotate": "Image Rotate",
    "mb Image Subtract": "Image Subtract",
    "mb Empty Latent Image": "Empty Latent Image (gpu)",
    "mb KSampler": "KSampler (gpu)",
    "mb KSampler Advanced": "KSampler Advanced (gpu)",
}
