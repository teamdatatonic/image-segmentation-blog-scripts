import numpy as np
import PIL
from google.cloud import aiplatform


def colour_mask(mask_np):
    id_to_colour = {
        0: PIL.ImageColor.getrgb("white"),
        1: PIL.ImageColor.getrgb("grey"),
        2: PIL.ImageColor.getrgb("red"),
        3: PIL.ImageColor.getrgb("yellow"),
        4: PIL.ImageColor.getrgb("green"),
        5: PIL.ImageColor.getrgb("blue"),
        6: PIL.ImageColor.getrgb("black"),
        7: PIL.ImageColor.getrgb("pink"),
    }
    colourised = np.zeros((mask_np.shape[0], mask_np.shape[1], 3))
    for (i, j), _ in np.ndenumerate(mask_np):
        val = mask_np[i, j]
        colourised[i, j, :] = id_to_colour[val]
    colourised = colourised.astype(np.uint8)
    return PIL.Image.fromarray(colourised)


project_id = "my-project"
region = "europe-west1"
bucket_name = "my_bucket"
endpoint_id = "9876543210"
image_path = "test.png"

aiplatform.init(project=project_id, location=region, staging_bucket=bucket_name)

endpoint = aiplatform.Endpoint(
    endpoint_name=endpoint_id,
    project=project_id,
    location=region,
)

image = PIL.Image.open(image_path)
prediction_size = (160, 160)
x_test = [np.array(image.resize(prediction_size), dtype="float32").tolist()]
predictions = endpoint.predict(instances=x_test)
mask_np = np.array(predictions.predictions[0])
mask_np = np.argmax(mask_np, axis=-1)
mask = colour_mask(mask_np)
mask = mask.resize(image.size)
mask.save("mask.png")
mask = mask.convert("RGBA")
mask.putalpha(128)
image = image.convert("RGBA")
overlay = PIL.Image.alpha_composite(image, mask)
overlay.save("overlay.png")
