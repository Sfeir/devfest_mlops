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
    """Creates a TFX pipeline using BigQuery."""

    # query data in BigQuery as a data source.
    output = tfx.proto.Output(
        # TODO: define custom split configuration with 80% train and 20% split data
        # reference: https://www.tensorflow.org/tfx/guide/examplegen#custom_inputoutput_split
    )

    example_gen = tfx.extensions.google_cloud_big_query.BigQueryExampleGen(
        query=query, output_config=output)

    # compute the statistics
    statistics_gen = tfx.components.StatisticsGen(
        # TODO: define StatisticsGen arguments
        # reference: https://www.tensorflow.org/tfx/guide/statsgen
    )

    # generate schema
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'])

    # pre-processe data
    transformer = tfx.components.Transform(
        # TODO: define Transform arguments
        # reference: https://www.tensorflow.org/tfx/guide/transform
    )

    # use user-provided Python function that trains a model
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
    serving_image = 'europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest'

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
                                             pushed_model=pusher.outputs['pushed_model'])

    components = [
        example_gen,
        # TODO: list the pipeline components
    ]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        beam_pipeline_args=beam_pipeline_args)
