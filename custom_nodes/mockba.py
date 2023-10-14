import os
import folder_paths
import gc
import torch
import base64
import hashlib
import numpy as np

from PIL import Image, ImageOps
from pprint import pprint

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management

import latent_preview

MAX_RESOLUTION = 8192


# A proxy class that always returns True when compared to any other object.
class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


# Flips an image horizontally or vertically.
class mbImageFlip:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "flip": (["none", "horizontal", "vertical", "both"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "flip_image"
    CATEGORY = "Mockba"
    DESCRIPTION = "Flips an image horizontally or vertically."

    def flip_image(self, image, flip):
        if flip == "none":
            return (image,)

        image_np = 255.0 * image.cpu().numpy().squeeze()

        if flip == "horizontal":
            flipped_image_np = np.flip(image_np, axis=1)
        elif flip == "vertical":
            flipped_image_np = np.flip(image_np, axis=0)
        elif flip == "both":
            flipped_image_np = np.flip(np.flip(image_np, axis=1), axis=0)
        else:
            print(f"Invalid flip. Must be either 'none', 'horizontal', 'vertical' or 'both'. No changes applied.")
            return (image,)

        flipped_image_np = flipped_image_np.astype(np.float32) / 255.0
        flipped_image_tensor = torch.from_numpy(flipped_image_np).unsqueeze(0)

        return (flipped_image_tensor,)


# Rotates an image by 90, 180 or 270 degrees ccw.
class mbImageRot:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "degrees": (["0", "90", "180", "270"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "rotate_image"
    CATEGORY = "Mockba"
    DESCRIPTION = "Rotates an image by 90, 180 or 270 degrees ccw."

    def rotate_image(self, image, degrees):
        if degrees == "0":
            return (image,)

        # Convert the input image tensor to a NumPy array
        image_np = 255.0 * image.cpu().numpy().squeeze()

        if degrees == "90":
            rotated_image_np = np.rot90(image_np)
        elif degrees == "180":
            rotated_image_np = np.rot90(image_np, 2)
        elif degrees == "270":
            rotated_image_np = np.rot90(image_np, 3)
        else:
            print(f"Invalid degrees. Must be either '0', '90', '180' or '270'. No changes applied.")
            return (image,)

        # Convert the rotated NumPy array back to a tensor
        rotated_image_np = rotated_image_np.astype(np.float32) / 255.0
        rotated_image_tensor = torch.from_numpy(rotated_image_np).unsqueeze(0)

        return (rotated_image_tensor,)


# Subtracts two images. Used to visually measure the difference between two images.
class mbImageSubtract:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": ("IMAGE",),
                "b": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "subtract"
    CATEGORY = "Mockba"
    DESCRIPTION = "Subtracts two images. Used to measure the difference between two images."

    def subtract(self, a, b):
        return (abs(a - b),)


# Returns the width and height of an image.
class mbImageDimensions:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("width", "height",)
    FUNCTION = "get_size"
    CATEGORY = "Mockba"
    DESCRIPTION = "Returns the width and height of an image."

    def get_size(self, image):
        image_size = image.size()
        image_width = int(image_size[2])
        image_height = int(image_size[1])
        return (image_width, image_height,)


# Loads an image from a file.
class mbImageLoad:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        exclude_folders = ["clipspace"]
        file_list = []

        for root, dirs, files in os.walk(input_dir):
            # Exclude specific folders
            dirs[:] = [d for d in dirs if d not in exclude_folders]
            
            for file in files:
                file_list.append(os.path.relpath(os.path.join(root, file), start=input_dir))

        return {"required":
                    {"image": (sorted(file_list), {"image_upload": True})},
                }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "load_image"
    CATEGORY = "Mockba"
    DESCRIPTION = "Loads image with subfolders."

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask.unsqueeze(0))

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


# Selects one of two objects based on the value of a slider.
class mbSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": (AlwaysEqualProxy("*"), {"default": 0}),
                "b": (AlwaysEqualProxy("*"), {"default": 0}),
                "select": (["a", "b"], {"default": "a"}),
            }
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("out",)
    FUNCTION = "select"
    CATEGORY = "Mockba"
    DESCRIPTION = "Selects one of two objects based on the value of a slider."

    def select(self, a, b, select):
        if select == "a":
            return (a,)
        else:
            return (b,)


# Execute python code on inputs and return the result.
class mbExecute:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": (AlwaysEqualProxy("*"), {"default": 0}),
                "code": ("STRING", {"default": ""}),
            },
            "optional": {
                "b": (AlwaysEqualProxy("*"), {"default": 0}),
            },
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("out",)
    FUNCTION = "execute"
    CATEGORY = "Mockba"
    DESCRIPTION = "Execute python code on input(s) and return the result."

    def execute(self, a, code, b=None):
        if code == "": return (a,)
        return (eval(code),)


# Saves an image to a file.
class mbImageToFile:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "base_name": ("STRING", {"default": "image"}),
                "id": ("INT", {"default": 0, "min": 0, "step": 1}),
                "use_id": (["yes", "no"], {"default": "no"}),
                "log": (["yes", "no"], {"default": "no"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("image", "id",)
    FUNCTION = "mbImageSave"
    CATEGORY = "Mockba"
    DESCRIPTION = "Saves an image to a file."

    def mbImageSave(self, image, base_name, id, use_id, log):
        prefix = "ComfyUI\\input\\"
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        if use_id == "yes":
            filename = base_name + "_" + str(id) + ".png"
        else:
            filename = base_name + ".png"
        image_np = 255.0 * image.cpu().numpy().squeeze()
        image_pil = Image.fromarray(image_np.astype(np.uint8))
        image_pil.save(prefix + filename)
        if log == "yes":
            print("Saved file: " + prefix + filename)
        return (image, id,)


# Loads an image from a file.
class mbFileToImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_name": ("STRING", {"default": "image"}),
                "id": ("INT", {"default": 0, "min": 0, "step": 1}),
                "use_id": (["yes", "no"], {"default": "no"}),
                "log": (["yes", "no"], {"default": "no"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("image", "id",)
    FUNCTION = "mbImageLoad"
    CATEGORY = "Mockba"
    DESCRIPTION = "Loads an image from a file."

    def mbImageLoad(self, base_name, id, use_id, log):
        prefix = "ComfyUI\\input\\"
        if use_id == "yes":
            filename = base_name + "_" + str(id) + ".png"
        else:
            filename = base_name + ".png"
        if not os.path.exists(prefix + filename):
            if log == "yes":
                print("File not found: " + prefix + filename)
            return (torch.zeros([1, 512, 512, 3]),)
        image_pil = Image.open(prefix + filename)
        image_np = np.array(image_pil).astype(np.float32) / 255.0
        image = torch.from_numpy(image_np).unsqueeze(0)
        if log == "yes":
            print("Loaded file: " + prefix + filename)
        return (image, id,)


# Saves text to a file.
class mbTextToFile:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "text"}),
                "base_name": ("STRING", {"default": "text"}),
                "id": ("INT", {"default": 0, "min": 0, "step": 1}),
                "use_id": (["yes", "no"], {"default": "no"}),
                "log": (["yes", "no"], {"default": "no"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("text", "id",)
    FUNCTION = "mbTextSave"
    CATEGORY = "Mockba"
    DESCRIPTION = "Saves text to a file."

    def mbTextSave(self, text, base_name, id, use_id, log):
        prefix = "ComfyUI\\input\\"
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        if use_id == "yes":
            filename = base_name + "_" + str(id) + ".txt"
        else:
            filename = base_name + ".txt"
        with open(prefix + filename, "w") as f:
            f.write(text)
        if log == "yes":
            print("Saved file: " + prefix + filename)
        return (text, id,)


# Loads text from a file.
class mbFileToText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "default": ("STRING", {"default": ""}),
                "base_name": ("STRING", {"default": "text"}),
                "id": ("INT", {"default": 0, "min": 0, "step": 1}),
                "use_id": (["yes", "no"], {"default": "no"}),
                "log": (["yes", "no"], {"default": "no"}),
            }
        }

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("text", "id",)
    FUNCTION = "mbTextLoad"
    CATEGORY = "Mockba"
    DESCRIPTION = "Loads text from a file."

    def mbTextLoad(self, default, base_name, id, use_id, log):
        if default != "":
            return (default, id)
        prefix = "ComfyUI\\input\\"
        if use_id == "yes":
            filename = base_name + "_" + str(id) + ".txt"
        else:
            filename = base_name + ".txt"
        if not os.path.exists(prefix + filename):
            if log == "yes":
                print("File not found: " + prefix + filename)
            return ("",)
        with open(prefix + filename, "r") as f:
            text = f.read()
        if log == "yes":
            print("Loaded file: " + prefix + filename)
        return (text, id,)


# loads text from a file or uses the entered value.
class mbTextOrFile:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"default": "", "multiline": True}),
                "base_name": ("STRING", {"default": "filename"}),
                "action": (["append", "prepend", "replace"], {"default": "append"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "mbTextOrFile"
    CATEGORY = "Mockba"
    DESCRIPTION = "Loads text from a file or uses the entered value."

    def mbTextOrFile(self, input, base_name, action):
        prefix = "ComfyUI\\input\\"
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        filename = base_name + ".txt"
        if not os.path.exists(prefix + filename):
            return (input,)
        with open(prefix + filename, "r") as f:
            file_text = f.read()
        if action == "append":
            file_text = file_text + input
        elif action == "prepend":
            file_text = input + file_text
        elif action == "replace":
            file_text = input
        return (file_text,)


# Shows debug information about the input object.
class mbDebug:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"input": (AlwaysEqualProxy("*"), {})},
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "debug"
    CATEGORY = "Mockba"
    DESCRIPTION = "Shows debug information about the input object."
    OUTPUT_NODE = True

    def debug(self, input):
    
        print("Debug:")
        print(input)
        if isinstance(input, object) and not isinstance(input, (str, int, float, bool, list, dict, tuple)):
            print("Objects directory listing:")
            pprint(dir(input), indent=4)
		
        return ()


# Creates an empty latent image using the cpu or gpu.
class mbEmptyLatentImage:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        devices = ["cpu", "cuda"]
        default_device = devices[0]

        resolutions = []
        with open("ComfyUI\\mockba\\resolutions.txt", "r") as f:
            for line in f:
                resolutions.append(line.strip())
        default_resolution = resolutions[0]

        return {
            "required": {
                "size": (
                    resolutions,
                    {"default": default_resolution},
                ),
                "width": (
                    "INT",
                    {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "device": (devices, {"default": default_device}),
                "garbage_collect": ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_image",)
    FUNCTION = "generate"
    CATEGORY = "Mockba"
    DESCRIPTION = "Creates an empty latent image using the cpu or gpu."

    def generate(self, size, width, height, batch_size=1, device="cpu", garbage_collect=False):
        self.device = device
        if garbage_collect:
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()


        if size == "------":
            size = "custom"
        if size == "custom":
            n_width = width
            n_height = height
        else:
            n_width = int(size.split("x")[0])
            n_height = int(size.split("x")[1])

        latent = torch.zeros(
            [batch_size, 4, n_height // 8, n_width // 8], device=device
        )
        return ({"samples": latent,},)


# creates random noise given a latent image and a seed.
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
    return (noises,)


# Runs a model with a given latent image using cpu or gpu and returns the resulting latent image.
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


# Runs a model with a given latent image using cpu or gpu and returns the resulting latent image.
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
    RETURN_NAMES = ("latent_image",)
    FUNCTION = "sample"
    CATEGORY = "Mockba"
    DESCRIPTION = "Runs a model with a given latent image using cpu or gpu and returns the resulting latent image."

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
        return (my_common_ksampler(
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
        ))


# Runs a model with a given latent image using cpu or gpu and returns the resulting latent image.
class mbKSamplerAdvanced:
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
    RETURN_NAMES = ("latent_image",)
    FUNCTION = "sample"
    CATEGORY = "Mockba"
    DESCRIPTION = "Runs a model with a given latent image using cpu or gpu and returns the resulting latent image."

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
        return (my_common_ksampler(
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
            force_full_denoise=force_full_denoise
        ))

# Generates a hash given a seed and a base string.
class mbHashGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("STRING", {"default": "000000000000"}),
                "base_string": ("STRING", {"default": "text"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("hash",)
    FUNCTION = "mbHashGenerate"
    CATEGORY = "Mockba"
    DESCRIPTION = "Generates a hash given a seed and a base string."

    def mbHashGenerate(self, seed, base_string):
        mac = seed.replace(':', '')
        dic = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        data = base64.b64decode("R2FyeQ==").decode('utf-8')
        data += mac
        data += base64.b64decode("bWFzdGVy").decode('utf-8')
        data += base_string
        md = hashlib.sha1(data.encode('utf-8')).digest()
        v = []
        for i in range(8):
            k = i + 8 + (i < 4) * 8
            v.append(md[i] ^ md[k])

        pw = ""
        for i in range(8):
            pw += dic[v[i] + 2 * (v[i] // 62) - ((v[i] // 62) << 6)]

        return(base_string+'-'+pw,)


# Maps node class names to their corresponding class.
NODE_CLASS_MAPPINGS = {
    "mb Image Flip": mbImageFlip,
    "mb Image Rotate": mbImageRot,
    "mb Image Subtract": mbImageSubtract,
    "mb Image Dimensions": mbImageDimensions,
    "mb Image Load": mbImageLoad,
    "mb Image to File": mbImageToFile,
    "mb File to Image": mbFileToImage,
    "mb Text to File": mbTextToFile,
    "mb File to Text": mbFileToText,
    "mb Text or File": mbTextOrFile,
    "mb Debug": mbDebug,
    "mb Selector": mbSelector,
    "mb Execute": mbExecute,
    "mb Empty Latent Image": mbEmptyLatentImage,
    "mb KSampler": mbKSampler,
    "mb KSampler Advanced": mbKSamplerAdvanced,
    "mb Hash Generator": mbHashGenerator,
}


# Maps node class names to their corresponding display names.
NODE_DISPLAY_NAME_MAPPINGS = {
    "mb Image Flip": "Image Flip",
    "mb Image Rotate": "Image Rotate",
    "mb Image Subtract": "Image Subtract",
    "mb Image Dimensions": "Image Dimensions",
    "mb Image Load": "Image Load 🖖",
    "mb Image to File": "Image to File",
    "mb File to Image": "File to Image",
    "mb Text to File": "Text to File",
    "mb File to Text": "File to Text",
    "mb Text or File": "Text or File",
    "mb Debug": "Debug",
    "mb Selector": "Selector",
    "mb Execute": "Execute",
    "mb Empty Latent Image": "Empty Latent Image (gpu) 🖖",
    "mb KSampler": "KSampler (gpu) 🖖",
    "mb KSampler Advanced": "KSampler Advanced (gpu) 🖖",
    "mb Hash Generator": "Hash Generator",
}
