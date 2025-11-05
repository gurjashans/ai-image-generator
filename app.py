from flask import Flask, request, jsonify, send_file
from diffusers import StableDiffusionPipeline
import torch, os

app = Flask(__name__)

# Load model (light version to fit free Render CPU)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cpu")

@app.route("/")
def home():
    return "🚀 AI Image Generator is Online! Use POST /generate with JSON {'prompt': 'your text'}"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "A futuristic city skyline at sunset")
    image = pipe(prompt).images[0]
    path = "output.png"
    image.save(path)
    return send_file(path, mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
