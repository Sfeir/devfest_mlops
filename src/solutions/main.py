import kfp
import os
import logging

import click

import tensorflow as tf
from tfx import v1 as tfx
from create_pipeline import _create_pipeline

from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs


@click.command()
@click.option("--google_cloud_project", required=True, type=click.STRING)
@click.option("--google_cloud_region", required=True, type=click.STRING)
@click.option("--google_cloud_dataset", required=True, type=click.STRING)
@click.option("--wine_table", required=True, type=click.STRING)
@click.option("--gcs_bucket", required=True, type=click.STRING)
@click.option("--username", required=True, type=click.STRING)
def main(google_cloud_project: str, google_cloud_region: str, google_cloud_dataset: str, wine_table: str,
         gcs_bucket: str, username: str):
    logging.getLogger().setLevel(logging.INFO)

    pipeline_name = f'wine-quality-{username}'
    pipeline_root = f'gs://{gcs_bucket}/pipeline_root/{pipeline_name}'
    module_root = f'gs://{gcs_bucket}/pipeline_module/{pipeline_name}'
    endpoint_name = 'prediction-' + pipeline_name
    sql_query = f"SELECT * FROM `{google_cloud_project}.{google_cloud_dataset}.{wine_table}`"

    big_query_with_direct_runner_beam_pipeline_args = [
        '--project=' + google_cloud_project,
        '--temp_location=' + os.path.join('gs://', gcs_bucket, 'tmp'),
    ]
    pipeline_definition_file = pipeline_name + '_pipeline.json'

    custom_tfx_image = f'europe-docker.pkg.dev/{google_cloud_project}/eu.gcr.io/tfx_augm:1.9.1'
    _transformer_module_file = 'transformer.py'
    _trainer_module_file = 'trainer.py'

    runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
        config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(default_image=custom_tfx_image),
        output_filename=pipeline_definition_file)

    _ = runner.run(
        _create_pipeline(
            pipeline_name=pipeline_name,
            pipeline_root=pipeline_root,
            query=sql_query,
            transformer_module_file=os.path.join(_transformer_module_file),
            trainer_module_file=os.path.join(_trainer_module_file),
            endpoint_name=endpoint_name,
            project_id=google_cloud_project,
            region=google_cloud_region,
            beam_pipeline_args=big_query_with_direct_runner_beam_pipeline_args))

    aiplatform.init(project=google_cloud_project, location=google_cloud_region)

    job = pipeline_jobs.PipelineJob(template_path=pipeline_definition_file,
                                    display_name=pipeline_name,
                                    enable_caching=False)
    job.submit()


if __name__ == "__main__":
    main()
