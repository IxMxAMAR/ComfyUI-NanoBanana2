# ComfyUI-NanoBanana2

A custom node for ComfyUI that integrates Google's Gemini models (including Gemini 3.5 Pro and Gemini 3.1 Flash Image) for advanced prompt refinement and high-fidelity multi-reference image generation. 

Specifically optimized for fashion, editorial detail, and precise structural consistency in ComfyUI workflows.

## Installation

1. Clone this repository into your `ComfyUI/custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/IxMxAMAR/ComfyUI-NanoBanana2.git
   ```
2. Ensure you have the Google GenAI SDK installed in your ComfyUI Python environment:
   ```bash
   pip install google-genai pillow numpy torch
   ```
3. Restart ComfyUI.

## Usage
Set your Gemini API Key in the node, or as an environment variable `GEMINI_API_KEY`.
