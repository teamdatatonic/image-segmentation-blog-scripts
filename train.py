from datetime import datetime

from google.cloud import aiplatform

project_id = "my-project"
region = "europe-west1"
bucket_name = "my_bucket"

aiplatform.init(project=project_id, location=region, staging_bucket=bucket_name)

train_image = "europe-docker.pkg.dev/vertex-ai/training/tf-gpu.2-8:latest"
deploy_image = "europe-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-8:latest"
accelerator_type = "NVIDIA_TESLA_K80"
script_path = "model.py"

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

job = aiplatform.CustomTrainingJob(
    display_name="custom_training_job-" + timestamp,
    script_path=script_path,
    container_uri=train_image,
    model_serving_container_image_uri=deploy_image,
)

epochs = 15
cmdargs = ["--bucket_name=" + args.bucket_name, "--epochs=" + str(epochs)]

job.run(
    model_display_name="my_model-" + timestamp,
    args=cmdargs,
    replica_count=1,
    machine_type="n1-standard-4",
    accelerator_type=accelerator_type,
    accelerator_count=1,
)
