from typing import Any, Dict, List
import tensorflow_data_validation as tfdv

from tfx import types
from tfx.dsl.components.base import base_executor

from google.cloud import aiplatform
from google.protobuf.duration_pb2 import Duration
from google.cloud.aiplatform_v1.services.job_service import JobServiceClient
from google.cloud.aiplatform_v1.types import (SamplingStrategy, ModelMonitoringObjectiveConfig,
                                              ModelMonitoringAlertConfig, ThresholdConfig,
                                              ModelDeploymentMonitoringScheduleConfig,
                                              ModelDeploymentMonitoringObjectiveConfig,
                                              ModelDeploymentMonitoringJob,
                                              ListModelDeploymentMonitoringJobsRequest
                                              )


def _get_features_names(features_file_uri: str):
    """
    Reads and parses json file containing features names from GCS bucket
    :param features_file_uri: URI pointing to data statistics file
    :return: statistics data
    """

    def load_statistics(file_path: str):
        return tfdv.load_stats_binary(file_path)

    return load_statistics(features_file_uri + '/Split-train/FeatureStats.pb')


class Executor(base_executor.BaseExecutor):
    """Executor for Monitorer Component."""

    def Do(self, input_dict: Dict[str, List[types.Artifact]],
           output_dict: Dict[str, List[types.Artifact]],
           exec_properties: Dict[str, Any]) -> None:
        """
        Launches a monitoring job for deployed on an endpoint model.
        :param input_dict: dictionary with input artifacts
            - statistics: statistics generated based on input data
            - pushed_model: deployed model
        :param output_dict: dictionary with output artifacts
        :param exec_properties: parameters passed in the call to create an instance of this component
            - project_id: id of Google Cloud project
            - region: Google Cloud region in which the model was deployed
            - default_threshold_value: default threshold value for data drift
            - monitoring_frequency: frequency at which model's recently logged inputs are monitored
            - sample_rate: percentage of input data logged
            - emails: emails used for notification
        """

        pushed_model_destination_uri = input_dict['pushed_model'][0].get_string_custom_property('pushed_destination')
        features_file_uri = input_dict['statistics'][0].uri

        project_id = exec_properties['project_id']
        region = exec_properties['region']

        aiplatform.init(project=project_id, location=region)

        model = aiplatform.Model(pushed_model_destination_uri)

        model_name = model.display_name
        endpoint_id = model.to_dict()['deployedModels'][0]['endpoint']
        deployed_model_id = model.to_dict()['deployedModels'][0]['deployedModelId']

        stats = _get_features_names(features_file_uri)
        api_vertex_endpoint = f"{region}-aiplatform.googleapis.com"

        features = {}
        for feature in stats.datasets[0].features:
            features[feature.path.step[0]]: {'mean': feature.num_stats.mean, 'std_dev': feature.num_stats.std_dev}

        sampling_config = SamplingStrategy.RandomSampleConfig(sample_rate=exec_properties['sample_rate'])
        sampling_strategy = SamplingStrategy(random_sample_config=sampling_config)

        monitoring_duration = Duration(seconds=exec_properties['monitoring_frequency'])
        monitoring_config = ModelDeploymentMonitoringScheduleConfig(monitor_interval=monitoring_duration)

        email_config = ModelMonitoringAlertConfig.EmailAlertConfig(user_emails=[exec_properties['emails']])
        alert_config = ModelMonitoringAlertConfig(email_alert_config=email_config)

        # monitoring whether feature data distribution changes significantly over time
        drift_thresholds = {}
        default_threshold = ThresholdConfig(value=exec_properties['default_threshold_value'])

        # set thresholds for all features
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
        # bq://<project_id>.model_deployment_monitoring_<endpoint_id>.<tolower(log_source)>_<tolower(log_type)>
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
        monitoring_job_search_response = client.list_model_deployment_monitoring_jobs(request=monitoring_job_request)

        response = client.create_model_deployment_monitoring_job(
            parent=parent, model_deployment_monitoring_job=monitoring_job
        )
