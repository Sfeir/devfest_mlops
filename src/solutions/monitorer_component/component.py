from typing import Optional

from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from monitorer_component import executor
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter


class MonitorerComponentSpec(types.ComponentSpec):
    """ComponentSpec for Custom TFX Hello World Component."""

    PARAMETERS = {
        # These are parameters that will be passed in the call to
        # create an instance of this component.
        'DEFAULT_THRESHOLD_VALUE': ExecutionParameter(type=float),
        'MONITORING_FREQUENCY': ExecutionParameter(type=int),
        'SAMPLE_RATE': ExecutionParameter(type=int),
        'EMAILS': ExecutionParameter(type=str)
    }
    INPUTS = {
        # This will be a dictionary with input artifacts, including URIs
        'statistics': ChannelParameter(type=standard_artifacts.ExampleStatistics),
        'pushed_model': ChannelParameter(type=standard_artifacts.PushedModel),
    }
    OUTPUTS = {
        # This will be a dictionary which this component will populate
        # 'output_data': ChannelParameter(type=standard_artifacts.Examples),
    }


class MonitorerComponent(base_component.BaseComponent):
    """Custom TFX Monitorer Component."""

    SPEC_CLASS = MonitorerComponentSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self,
                 statistics: types.Channel = None,
                 pushed_model: types.Channel = None,
                 DEFAULT_THRESHOLD_VALUE: Optional[float] = 0.03,
                 MONITORING_FREQUENCY: Optional[int] = 3600,
                 SAMPLE_RATE: Optional[int] = 0,
                 EMAILS: Optional[str] = "",
                 ):
        # statistics_artifact = standard_artifacts.ExampleStatistics()
        # pushed_model_artifact = standard_artifacts.PushedModel()

        # statistics_artifact.split_names = statistics.get()[0].split_names
        # pushed_model_artifact.split_names = pushed_model.get()[0].split_names

        spec = MonitorerComponentSpec(statistics=statistics,
                                      pushed_model=pushed_model,
                                      DEFAULT_THRESHOLD_VALUE=DEFAULT_THRESHOLD_VALUE,
                                      MONITORING_FREQUENCY=MONITORING_FREQUENCY,
                                      SAMPLE_RATE=SAMPLE_RATE,
                                      EMAILS=EMAILS
                                      )
        super(MonitorerComponent, self).__init__(spec=spec)
