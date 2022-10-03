
import tensorflow as tf
import tensorflow_transform as tft

from typing import List
from absl import logging

from tensorflow import keras
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata import schema_utils

from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from keras_tuner.engine import base_tuner

_FEATURE_KEYS = [
    'fixed_acidity',	'volatile_acidity',	'citric_acid',	'residual_sugar',
    'chlorides',	'free_sulfur_dioxide',	'total_sulfur_dioxide',	'density',
    'ph',	'sulphates',	'alcohol'
]
_LABEL_KEY = 'quality'

_TRAIN_BATCH_SIZE = 100
_EVAL_BATCH_SIZE = 20


def _input_fn(file_pattern: List[str],
              tf_transform_output,
              data_accessor: tfx.components.DataAccessor,
              batch_size: int) -> tf.data.Dataset:

  return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(batch_size=batch_size, label_key=_LABEL_KEY),
      schema=tf_transform_output.transformed_metadata.schema).repeat()


def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example and applies TFT."""

  # Get transformation graph
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)

    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    transformed_features = model.tft_layer(parsed_features)

    return model(transformed_features)

  return serve_tf_examples_fn


def _make_keras_model() -> tf.keras.Model:
  """Creates a DNN Keras model for classifying penguin data.

  Returns:
    A Keras Model.
  """
  # The model below is built with Functional API, please refer to
  # https://www.tensorflow.org/guide/keras/overview for all API options.
  inputs = [keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]

  model_layers = keras.layers.concatenate(inputs)
  model_layers = keras.layers.Dropout(0.2)(model_layers)
  model_layers = keras.layers.BatchNormalization()(model_layers)
  model_layers = keras.layers.Dense(units=8, activation='relu')(model_layers)
  model_layers = keras.layers.Dropout(0.2)(model_layers)
  model_layers = keras.layers.BatchNormalization()(model_layers)
  model_layers = keras.layers.Dense(units=14, activation='relu')(model_layers)
  model_layers = keras.layers.Dropout(0.2)(model_layers)
  model_layers = keras.layers.BatchNormalization()(model_layers)
  model_layers = keras.layers.Dense(units=6, activation='relu')(model_layers)
  model_layers = keras.layers.Dropout(0.2)(model_layers)
  model_layers = keras.layers.BatchNormalization()(model_layers)

  outputs = keras.layers.Dense(1)(model_layers)

  model = keras.Model(inputs=inputs, outputs=outputs)

  rms_prop_optimizer = tf.keras.optimizers.RMSprop(learning_rate=10e-5)
  model.compile(
      optimizer=rms_prop_optimizer,
      loss=tf.keras.losses.binary_crossentropy,
      metrics=['accuracy']
  )

  model.summary(print_fn=logging.info)
  return model


TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
  """Build the tuner using the KerasTuner API.
  Args:
    fn_args: Holds args as name/value pairs.
      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.
  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """
  tuner = Tuner(
    module_file=module_file,  # Contains `tuner_fn`.
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    train_args=trainer_pb2.TrainArgs(num_steps=20),
    eval_args=trainer_pb2.EvalArgs(num_steps=5))

    trainer = Trainer(
    module_file=module_file,  # Contains `run_fn`.
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    # This will be passed to `run_fn`.
    hyperparameters=tuner.outputs['best_hyperparameters'],
    train_args=trainer_pb2.TrainArgs(num_steps=100),
    eval_args=trainer_pb2.EvalArgs(num_steps=5))


def run_fn(fn_args: tfx.components.FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(
      fn_args.train_files,
      tf_transform_output,
      fn_args.data_accessor,
      batch_size=_TRAIN_BATCH_SIZE)
  eval_dataset = _input_fn(
      fn_args.eval_files,
      tf_transform_output,
      fn_args.data_accessor,
      batch_size=_EVAL_BATCH_SIZE)

  model = _make_keras_model()
  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)
  
  # Define default serving signature
  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples'))
  }

  # The result of the training should be saved in `fn_args.serving_model_dir`
  # directory.
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
