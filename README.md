# Image Segmentation Blog Scripts
Repository containing the code snippets included in Datatonic's [image segmentation blog](https://datatonic.com/insights/deploying-image-segmentation-models-vertex-ai).
Trains an image segmentation model using Vertex AI custom training and deploys it to a Vertex AI endpoint.
Uses a [U-Net model](https://arxiv.org/pdf/1505.04597.pdf) and the [Cityscapes dataset](https://www.cityscapes-dataset.com/).

Based on:
- [Custom training and online prediction](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/custom/sdk-custom-image-classification-online.ipynb)
- [Image segmentation with a U-Net-like architecture](https://keras.io/examples/vision/oxford_pets_image_segmentation/)

## Dependencies
Install dependencies using [Poetry](https://python-poetry.org/).
