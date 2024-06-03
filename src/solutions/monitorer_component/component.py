from typing import Optional
from tfx import types

from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter

from monitorer_component import executor


class MonitorerComponentSpec(types.ComponentSpec):
    """ComponentSpec for Custom TFX Monitorer Component."""

    PARAMETERS = {
        'project_id': ExecutionParameter(type=str),
        'region': ExecutionParameter(type=str),
        'email': ExecutionParameter(type=str),
        'default_threshold_value': ExecutionParameter(type=float),
        'monitoring_frequency': ExecutionParameter(type=int),
        'sample_rate': ExecutionParameter(type=float)
    }
    INPUTS = {
        'statistics': ChannelParameter(type=standard_artifacts.ExampleStatistics),
        'pushed_model': ChannelParameter(type=standard_artifacts.PushedModel),
    }
    OUTPUTS = {
    }


class MonitorerComponent(base_component.BaseComponent):
    """Custom TFX Monitorer Component."""

    SPEC_CLASS = MonitorerComponentSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self,
                 statistics: types.Channel,
                 pushed_model: types.Channel,
                 project_id: str,
                 region: str,
                 email: str,
                 default_threshold_value: Optional[float] = 0.03,
                 monitoring_frequency: Optional[int] = 3600,
                 sample_rate: Optional[float] = 0.6,
                 ):
        """
        Construct a Monitorer Component
        :param statistics: statistics generated based on input data
        :param pushed_model: deployed model
        :param project_id: id of Google Cloud project
        :param region: Google Cloud region in which the model was deployed
        :param email: email used for notification
        :param default_threshold_value: default threshold value for data drift
        :param monitoring_frequency: frequency at which model's recently logged inputs are monitored
        :param sample_rate: percentage of input data logged
        """
        spec = MonitorerComponentSpec(statistics=statistics,
                                      pushed_model=pushed_model,
                                      project_id=project_id,
                                      region=region,
                                      email=email,
                                      default_threshold_value=default_threshold_value,
                                      monitoring_frequency=monitoring_frequency,
                                      sample_rate=sample_rate
                                      )
        super(MonitorerComponent, self).__init__(spec=spec)
