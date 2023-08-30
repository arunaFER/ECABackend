# API for image caption generator

# Imports
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel
import io
import json
import requests
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image


# Models
model = VisionEncoderDecoderModel.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set maximum caption length and number of beams for beam search
max_length = 16
num_beams = 4

# Set generation parameters for the model
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Define the predict_step function


def predict_step(image_paths):
    images = []

    # Load and preprocess each image
    for image_path in image_paths:
        i_image = image_path

        # Convert image to RGB mode if necessary
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    # Extract pixel values from the images using the feature extractor
    pixel_values = feature_extractor(
        images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate output captions for the images
    output_ids = model.generate(pixel_values, **gen_kwargs)

    # Decode the output IDs into text captions
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # Clean up the predicted captions
    preds = [pred.strip() for pred in preds]

    return preds


# Create FatAPI instance
app = FastAPI(title="Image Caption Generator API",
              description="The API for image caption generation using ViT-GPT2 model from HuggingFace library")


class ImageCaption(BaseModel):
    caption: str


# Endpoints

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse(url="/docs")


@app.post("/predict", response_model=ImageCaption)
def predict(file: UploadFile = File(...)):
    # Load img file to memory
    contents = file.file.read()
    image = Image.open(io.BytesIO(contents))
    result = predict_step([image])
    return JSONResponse(content={"caption": result})
