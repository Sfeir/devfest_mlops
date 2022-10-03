import json
import os
from absl import logging
from typing import Any, Dict, List

from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.dsl.io import fileio
from tfx.types import artifact_utils
from tfx.utils import io_utils


class Executor(base_executor.BaseExecutor):
    """Executor for MonitorerComponent."""

    def Do(self, input_dict: Dict[str, List[types.Artifact]],
           exec_properties: Dict[str, Any]) -> None:

        from google.cloud import aiplatform
        import tensorflow_data_validation as tfdv
        from google.cloud.aiplatform_v1.services.job_service import JobServiceClient
        from google.cloud.aiplatform_v1.types import (SamplingStrategy, ModelMonitoringObjectiveConfig,
                                                      ModelMonitoringAlertConfig, ThresholdConfig,
                                                      ModelDeploymentMonitoringScheduleConfig,
                                                      ModelDeploymentMonitoringObjectiveConfig,
                                                      ModelDeploymentMonitoringJob,
                                                      ListModelDeploymentMonitoringJobsRequest
                                                      )
        from google.protobuf.duration_pb2 import Duration

        pushed_model_destination_uri = input_dict['pushed_model'][0].get_string_custom_property('pushed_destination')
        features_file_uri = input_dict['statistics'][0].uri

        pushed_destination_list = pushed_model_destination_uri.split('/')
        project_id = pushed_destination_list[1]
        region = pushed_destination_list[3]

        aiplatform.init(project=project_id, location=region)

        model = aiplatform.Model(pushed_model_destination_uri)

        model_name = model.display_name
        endpoint_id = model.to_dict()['deployedModels'][0]['endpoint']
        deployed_model_id = model.to_dict()['deployedModels'][0]['deployedModelId']

        def get_features_names(features_file_uri: str):
            """
            Reads and parses json file containing features names from GCS bucket
            :param features_file_uri:
            :return:
            """

            def load_statistics(file_path: str):
                return tfdv.load_stats_binary(file_path)

            return load_statistics(features_file_uri + '/Split-train/FeatureStats.pb')

        stats = get_features_names(features_file_uri)
        api_vertex_endpoint = f"{region.upper()}-aiplatform.googleapis.com"

        features = {}
        for feature in stats.datasets[0].features:
            features[feature.path.step[0]]: {'mean': feature.num_stats.mean, 'std_dev': feature.num_stats.std_dev}

        sampling_config = SamplingStrategy.RandomSampleConfig(sample_rate=exec_properties['SAMPLE_RATE'])
        sampling_strategy = SamplingStrategy(random_sample_config=sampling_config)

        monitoring_duration = Duration(seconds=exec_properties['MONITORING_FREQUENCY'])
        monitoring_config = ModelDeploymentMonitoringScheduleConfig(monitor_interval=monitoring_duration)

        email_config = ModelMonitoringAlertConfig.EmailAlertConfig(user_emails=[exec_properties['EMAILS']])
        alert_config = ModelMonitoringAlertConfig(email_alert_config=email_config)

        # monitoring whether feature data distribution changes significantly over time
        drift_thresholds = {}
        default_threshold = ThresholdConfig(value=exec_properties['DEFAULT_THRESHOLD_VALUE'])

        # set thresholds as default for all features
        for feature in features:
            drift_thresholds[feature] = 1, 96 * features[feature]['std_dev'] + features[feature]['mean']

        drift_config = ModelMonitoringObjectiveConfig.PredictionDriftDetectionConfig(
            drift_thresholds=drift_thresholds
        )

        objective_config = ModelMonitoringObjectiveConfig(
            prediction_drift_detection_config=drift_config
        )
        monitoring_objective_configs = ModelDeploymentMonitoringObjectiveConfig(
            objective_config=objective_config
        )
        monitoring_objective_configs.deployed_model_id = deployed_model_id

        # create the monitoring job
        predict_schema = ""
        analysis_schema = ""

        # monitoring job will  create up to 4 bq tables with names :
        #   bq://<project_id>.model_deployment_monitoring_<endpoint_id>.<tolower(log_source)>_<tolower(log_type)>
        # https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform_v1beta1.types.ModelDeploymentMonitoringJob
        monitoring_job = ModelDeploymentMonitoringJob(
            display_name=f"monitoring_{model_name}",
            endpoint=endpoint_id,
            model_deployment_monitoring_objective_configs=[monitoring_objective_configs],
            logging_sampling_strategy=sampling_strategy,
            model_deployment_monitoring_schedule_config=monitoring_config,
            model_monitoring_alert_config=alert_config,
            predict_instance_schema_uri=predict_schema,
            analysis_instance_schema_uri=analysis_schema,
            enable_monitoring_pipeline_logs=True
        )
        parent = f"projects/{project_id}/locations/{region}"

        monitoring_job_request = ListModelDeploymentMonitoringJobsRequest(
            parent=parent, filter=f"display_name=monitoring_{model_name}"
        )

        options = dict(api_endpoint=api_vertex_endpoint)
        client = JobServiceClient(client_options=options)
        monitoring_job_search_response = client.list_model_deployment_monitoring_jobs(
            request=monitoring_job_request)

        if len([page for page in monitoring_job_search_response.pages])==0:
            logging.info('Monitoring job already active')
        else:
            response = client.create_model_deployment_monitoring_job(
                parent=parent, model_deployment_monitoring_job=monitoring_job
            )
