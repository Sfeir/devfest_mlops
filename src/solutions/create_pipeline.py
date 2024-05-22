from tfx import v1 as tfx
from typing import List, Optional

from monitorer_component import component


def _create_pipeline(pipeline_name: str, pipeline_root: str, query: str,
                     transformer_module_file: str,
                     trainer_module_file: str,
                     endpoint_name: str,
                     project_id: str, region: str,
                     beam_pipeline_args: Optional[List[str]],
                     ) -> tfx.dsl.Pipeline:
    """
    Creates a TFX pipeline with BigQuery as a source, trained a model, deploys it to Vertex AI endpoints
    and creates a monitoring job to analyze input requests for data drift.
    :param pipeline_name: name of the pipeline
    :param pipeline_root: Google Cloud Storage URI to store pipeline artifacts
    :param query: sql query to extract data
    :param transformer_module_file: custom transformation routine to pre-process data
    :param trainer_module_file: custom model training
    :param endpoint_name: name of an endpoint the trained model is going to be deployed to
    :param project_id: project id
    :param region: Google Cloud region
    :param beam_pipeline_args: project settings necessary for BigQuery query component
    :return TFX Pipeline
    """

    # query data in BigQuery as a data source
    output = tfx.proto.Output(
        split_config=tfx.proto.SplitConfig(splits=[
            tfx.proto.SplitConfig.Split(name='train', hash_buckets=4),
            tfx.proto.SplitConfig.Split(name='eval', hash_buckets=1)
        ]))

    example_gen = tfx.extensions.google_cloud_big_query.BigQueryExampleGen(
        query=query, output_config=output)

    # compute the statistics
    statistics_gen = tfx.components.StatisticsGen(examples=example_gen.outputs['examples'])

    # generate schema
    schema_gen = tfx.components.SchemaGen(statistics=statistics_gen.outputs['statistics'])

    # pre-process data
    transformer = tfx.components.Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=transformer_module_file)

    # train the model with user-provided Python function
    trainer = tfx.components.Trainer(
        module_file=trainer_module_file,
        examples=transformer.outputs['transformed_examples'],
        transform_graph=transformer.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=tfx.proto.TrainArgs(num_steps=1000),
        eval_args=tfx.proto.EvalArgs(num_steps=50))

    # push the model to model registry
    vertex_serving_spec = {
        'project_id': project_id,
        'endpoint_name': endpoint_name,
        'machine_type': 'n1-standard-4'
    }
    serving_image = 'europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest'

    pusher = tfx.extensions.google_cloud_ai_platform.Pusher(
        model=trainer.outputs['model'],
        custom_config={
            tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:
                True,
            tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:
                region,
            tfx.extensions.google_cloud_ai_platform.VERTEX_CONTAINER_IMAGE_URI_KEY:
                serving_image,
            tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY:
                vertex_serving_spec,
        })

    # create the monitoring job
    monitorer = component.MonitorerComponent(statistics=statistics_gen.outputs['statistics'],
                                             pushed_model=pusher.outputs['pushed_model'],
                                             project_id=project_id,
                                             region=region)

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        transformer,
        trainer,
        pusher,
        monitorer
    ]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        beam_pipeline_args=beam_pipeline_args)
