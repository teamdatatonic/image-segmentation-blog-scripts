from datetime import datetime

from google.cloud import aiplatform

project_id = "my-project"
region = "europe-west1"
bucket_name = "my_bucket"
model_id = "0123456789"
accelerator_type = "NVIDIA_TESLA_K80"

aiplatform.init(project=project_id, location=region, staging_bucket=bucket_name)

model = aiplatform.Model(model_name=model_id, project=project_id, location=region)

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

model.deploy(
    deployed_model_display_name="my_model-" + timestamp,
    traffic_split={"0": 100},
    machine_type="n1-standard-4",
    accelerator_type=accelerator_type,
    accelerator_count=1,
    min_replica_count=1,
    max_replica_count=1,
)
