import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# Hardcode the values of your computer vision endpoint and computer vision key
endpoint = "https://azurecomputervisionibad.cognitiveservices.azure.com/"
key = "f033518ca0e840faae6333c01aa82f8c"

# Create an Image Analysis client
client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

# Get a caption for the image. This will be a synchronously (blocking) call.
result = client.analyze_from_url(
    image_url="https://www.google.com/imgres?imgurl=https%3A%2F%2Ffiles.realpython.com%2Fmedia%2Fresults2.5d8d3b8108cd.png&tbnid=VtsyAgp79zyYnM&vet=12ahUKEwjml7zl5uqIAxWXAvsDHbnBBsYQMygGegQIARBR..i&imgrefurl=https%3A%2F%2Frealpython.com%2Fsetting-up-a-simple-ocr-server%2F&docid=C_63-j6ny4JIIM&w=455&h=262&q=images%20with%20urls%20for%20ocr%20on%20python&hl=en-GB&ved=2ahUKEwjml7zl5uqIAxWXAvsDHbnBBsYQMygGegQIARBR",
    visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ],
    gender_neutral_caption=True,  # Optional (default is False)
)

print("Image analysis results:")
# Print caption results to the console
print(" Caption:")
if result.caption is not None:
    print(f"   '{result.caption.text}', Confidence {result.caption.confidence:.4f}")

# Print text (OCR) analysis results to the console
print(" Read:")
if result.read is not None:
    for line in result.read.blocks[0].lines:
        print(f"   Line: '{line.text}', Bounding box {line.bounding_polygon}")
        for word in line.words:
            print(f"     Word: '{word.text}', Bounding polygon {word.bounding_polygon}, Confidence {word.confidence:.4f}")
