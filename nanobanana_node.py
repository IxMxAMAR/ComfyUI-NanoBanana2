import os
import torch
import numpy as np
import io
import time
from PIL import Image
from google import genai
from google.genai import types

class NanoBananaPromptRefiner:
    """
    A ComfyUI node that uses Gemini 3.5 Pro to refine and optimize 
    image generation prompts, specifically tailored for Christian Koehlert fashion.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False, 
                    "default": os.environ.get("GEMINI_API_KEY", "")
                }),
                "model": ([
                    "gemini-3.1-pro-preview", 
                    "gemini-3-pro-preview", 
                    "gemini-3.1-flash-lite-preview",
                    "gemini-2.5-pro", 
                    "gemini-2.5-flash", 
                    "gemini-2.0-flash", 
                    "nano-banana-pro-preview",
                    "deep-research-pro-preview-12-2025"
                ], {"default": "gemini-2.5-pro"}),
                "custom_model": ("STRING", {
                    "multiline": False, 
                    "default": ""
                }),
                "base_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "A woman wearing my dress in Paris."
                }),
                "system_prompts": ("STRING", {
                    "multiline": True, 
                    "default": ""
                }),
            },
            "optional": {}
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("refined_prompt",)
    FUNCTION = "refine_prompt"
    CATEGORY = "NanoBanana2"

    def refine_prompt(self, api_key, model, custom_model, base_prompt, system_prompts):
        if not api_key:
            raise ValueError("Gemini API Key is required.")
            
        final_model = custom_model.strip() if custom_model and custom_model.strip() != "" else model
        client = genai.Client(api_key=api_key, http_options={'timeout': 180000})
        
        # --- Build System Context ---
        system_context = (
            "You are a world-class Fashion Art Director and Prompt Engineer. "
            "Your task is to take a user's base concept and expand it into a highly detailed, "
            "professional prompt for an AI Image Generator.\n\n"
        )
        
        if system_prompts and system_prompts.strip():
            system_context += f"Refinement Style & Instructions: {system_prompts.strip()}"
            
        system_context += "\n\nCRITICAL: Output ONLY the final refined prompt text. Do not include introductory or concluding conversational text."
        
        # --- Generate Request ---
        retries = 3
        for attempt in range(retries):
            try:
                response = client.models.generate_content(
                    model=final_model,
                    contents=base_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_context,
                        temperature=0.7
                    )
                )
                return (response.text.strip(),)
            except Exception as e:
                is_transient = "504" in str(e) or "DEADLINE_EXCEEDED" in str(e) or "503" in str(e) or "429" in str(e) or "500" in str(e) or "502" in str(e)
                if is_transient and attempt < retries - 1:
                    sleep_time = 2 ** attempt * 5
                    print(f"NanoBanana2: API Error in prompt refiner ({e}), retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    raise e


class NanoBanana2MultiRef:
    """
    A ComfyUI node for multi-reference image generation using Gemini 3.1 Flash Image.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False, 
                    "default": os.environ.get("GEMINI_API_KEY", "")
                }),
                "model": ([
                    "imagen-4.0-ultra-generate-001", 
                    "imagen-4.0-generate-001", 
                    "imagen-4.0-fast-generate-001",
                    "gemini-3.1-flash-image-preview", 
                    "gemini-3-pro-image-preview", 
                    "gemini-2.5-flash-image"
                ], {"default": "imagen-4.0-generate-001"}),
                "custom_model": ("STRING", {
                    "multiline": False, 
                    "default": ""
                }),
                "prompt": ("STRING", {"multiline": True, "default": "[Reference Image 1]"}),
                "thinking_level": (["NONE", "HIGH", "NORMAL", "LOW"], {"default": "HIGH"}),
                "image_size": (["AUTO", "1K", "2K", "4K"], {"default": "4K"}),
                "aspect_ratio": (["AUTO", "1:1", "3:4", "4:3", "9:16", "16:9"], {"default": "16:9"}),
                "system_instruction": ("STRING", {
                    "multiline": True,
                    "default": "Act as a world-class Digital Art Director and Cinematographer. Your goal is to generate images with 100% structural fidelity, vibrant lighting, and zero 'hallucination drift'.\n1. ARCHITECTURAL LOGIC: Before rendering, analyze the physics of drapes, folds, and material intersections.\n2. SUBJECT CONSISTENCY: When reference images are provided, treat them as immutable templates. Assigned Subjects must maintain their identity.\n3. CINEMATIC LIGHTING: Unless specified, use a 'Three-point Softbox' or 'Golden Hour Editorial' setup.\n4. TEXTURE FIDELITY: Prioritize micro-details such as skin pores, fabric weave, beading sparkle, and hair strands."
                }),
            },
            "optional": {
                "reference_images_1": ("IMAGE",),
                "reference_images_2": ("IMAGE",),
                "reference_images_3": ("IMAGE",),
                "reference_images_4": ("IMAGE",),
                "reference_images_5": ("IMAGE",),
                "reference_images_6": ("IMAGE",),
                "reference_images_7": ("IMAGE",),
                "reference_images_8": ("IMAGE",),
                "reference_images_9": ("IMAGE",),
                "reference_images_10": ("IMAGE",),
                "text_file_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "NanoBanana2"

    def _tensor_to_bytes(self, tensor_image):
        """Converts a ComfyUI PyTorch Image tensor [B, H, W, C] to JPEG bytes to save bandwidth."""
        # We only take the first image if it's a batch
        i = 255. * tensor_image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        byte_stream = io.BytesIO()
        img.save(byte_stream, format='JPEG', quality=95)
        return byte_stream.getvalue()
        
    def _bytes_to_tensor(self, image_bytes):
        """Converts PNG bytes to a ComfyUI PyTorch Image tensor [B, H, W, C]."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(img).astype(np.float32) / 255.0
        # Add batch dimension
        image_tensor = torch.from_numpy(image_np)[None,]
        return image_tensor

    def generate_image(
        self, 
        api_key, 
        model, 
        custom_model,
        prompt, 
        thinking_level, 
        image_size, 
        aspect_ratio, 
        system_instruction,
        reference_images_1=None,
        reference_images_2=None,
        reference_images_3=None,
        reference_images_4=None,
        reference_images_5=None,
        reference_images_6=None,
        reference_images_7=None,
        reference_images_8=None,
        reference_images_9=None,
        reference_images_10=None,
        text_file_path=None
    ):
        if not api_key:
            raise ValueError("Gemini API Key is required.")
            
        final_model = custom_model.strip() if custom_model and custom_model.strip() != "" else model
        client = genai.Client(api_key=api_key, http_options={'timeout': 180000})
        parts = []
        
        # --- Process Reference Images ---
        ref_inputs = [
            reference_images_1, reference_images_2, reference_images_3, reference_images_4,
            reference_images_5, reference_images_6, reference_images_7, reference_images_8,
            reference_images_9, reference_images_10
        ]
        ref_count = 1
        
        for tensor_batch in ref_inputs:
            if tensor_batch is not None:
                # tensor_batch shape is usually [B, H, W, C] in ComfyUI
                try:
                    # In case of batch size > 1, we pull out individual images
                    for i in range(tensor_batch.shape[0]):
                        # slice to get [1, H, W, C]
                        single_tensor = tensor_batch[i:i+1] 
                        img_bytes = self._tensor_to_bytes(single_tensor)
                        
                        label_text = f"--- [Reference Image {ref_count}] ---"
                        parts.append(types.Part.from_text(text=label_text))
                        parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
                        
                        ref_count += 1
                except Exception as e:
                    print(f"Error processing reference image {ref_count}: {e}")

        # --- Main Prompt & Text Reference ---
        final_prompt = prompt
        if ref_count > 1:
            final_prompt = f"--- [Main Prompt] ---\n{final_prompt}"
            
        parts.append(types.Part.from_text(text=final_prompt))
        
        # --- Process Text/PDF File ---
        if text_file_path and text_file_path.strip() != "":
            try:
                clean_path = text_file_path.strip().strip('"').strip("'")
                if os.path.exists(clean_path):
                    ext = os.path.splitext(clean_path)[1].lower()
                    if ext == '.txt':
                        with open(clean_path, 'r', encoding='utf-8') as f:
                            text_content = f.read()
                        parts.append(types.Part.from_text(text=f"--- [Reference Text from {os.path.basename(clean_path)}] ---\n{text_content}"))
                    elif ext == '.pdf':
                        print(f"NanoBanana2: Uploading PDF to Gemini API: {clean_path}")
                        uploaded_file = client.files.upload(file=clean_path, mime_type="application/pdf")
                        parts.append(types.Part.from_uri(file_uri=uploaded_file.uri, mime_type="application/pdf"))
                    else:
                        print(f"NanoBanana2 Error: Unsupported file type for text_file_path: {ext}")
                else:
                    print(f"NanoBanana2 Error: The file {clean_path} does not exist.")
            except Exception as e:
                print(f"NanoBanana2 Error processing document: {e}")
                
        contents = [types.Content(role="user", parts=parts)]
        
        # --- Configuration ---
        sys_parts = []
        if system_instruction:
            sys_parts = [types.Part.from_text(text=system_instruction)]
            
        kwargs = {}
        if thinking_level and thinking_level != "NONE":
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
            
        image_kwargs = {}
        if aspect_ratio and aspect_ratio != "AUTO":
            image_kwargs["aspect_ratio"] = aspect_ratio
        if image_size and image_size != "AUTO":
            image_kwargs["image_size"] = image_size
            
        if image_kwargs:
            kwargs["image_config"] = types.ImageConfig(**image_kwargs)
            
        generate_content_config = types.GenerateContentConfig(
            response_modalities=[
                "IMAGE",
                "TEXT",
            ],
            system_instruction=sys_parts if sys_parts else None,
            **kwargs
        )
            
        # --- Execute API Call ---
        final_image_tensor = None
        
        retries = 3
        for attempt in range(retries):
            try:
                for chunk in client.models.generate_content_stream(
                    model=final_model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if chunk.parts is None:
                        continue
                    if chunk.parts[0].inline_data and chunk.parts[0].inline_data.data:
                        final_image_tensor = self._bytes_to_tensor(chunk.parts[0].inline_data.data)
                        break
                    elif chunk.text:
                        print(f"NanoBanana2 API Message: {chunk.text}")
                
                # If we successfully completed the chunk loop (or broke out of it), exit the retry loop
                break
            except Exception as e:
                is_transient = "504" in str(e) or "DEADLINE_EXCEEDED" in str(e) or "503" in str(e) or "429" in str(e) or "500" in str(e) or "502" in str(e)
                if is_transient and attempt < retries - 1:
                    sleep_time = 2 ** attempt * 5
                    print(f"NanoBanana2: API Error in image generation ({e}), retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    raise e
                    
        if final_image_tensor is None:
            raise RuntimeError("Gemini API stream completed but returned no image data.")
            
        return (final_image_tensor,)

# ComfyUI Node Exports
NODE_CLASS_MAPPINGS = {
    "NanoBananaPromptRefiner": NanoBananaPromptRefiner,
    "NanoBanana2MultiRef": NanoBanana2MultiRef
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaPromptRefiner": "Prompt Refiner (Gemini Pro)",
    "NanoBanana2MultiRef": "Nano Banana 2 Multi-Reference"
}
