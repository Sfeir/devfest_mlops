import tensorflow as tf
import tensorflow_transform as tft

from typing import List
from absl import logging
from tensorflow import keras
from tfx import v1 as tfx

_FEATURE_KEYS = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
    'ph', 'sulphates', 'alcohol'
]
_LABEL_KEY = 'quality'

_TRAIN_BATCH_SIZE = 512
_EVAL_BATCH_SIZE = 128
_NB_EPOCH = 200


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _input_fn(file_pattern: List[str],
              tf_transform_output,
              num_epochs: int,
              batch_size: int) -> tf.data.Dataset:
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    return tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=_LABEL_KEY)


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT."""

    # get transformation graph
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


def _get_serve_raw(model, transform_output):
    """Returns a function that parses a raw data request and applies TFT."""

    model.tft_layer = transform_output.transform_features_layer()

    @tf.function
    def serve_raw_fn(alcohol, chlorides, citric_acid, density, fixed_acidity, free_sulfur_dioxide,
                     ph, residual_sugar, sulphates, total_sulfur_dioxide, volatile_acidity):
        """Returns the output to be used in the serving signature."""

        alcohol_tensor = tf.dtypes.cast(alcohol, tf.float32)
        chlorides_tensor = tf.dtypes.cast(chlorides, tf.float32)
        citric_acid_tensor = tf.dtypes.cast(citric_acid, tf.float32)
        density_tensor = tf.dtypes.cast(density, tf.float32)
        fixed_acidity_tensor = tf.dtypes.cast(fixed_acidity, tf.float32)
        free_sulfur_dioxide_tensor = tf.dtypes.cast(free_sulfur_dioxide, tf.float32)
        ph_tensor = tf.dtypes.cast(ph, tf.float32)
        residual_sugar_tensor = tf.dtypes.cast(residual_sugar, tf.float32)
        sulphates_tensor = tf.dtypes.cast(sulphates, tf.float32)
        total_sulfur_dioxide_tensor = tf.dtypes.cast(total_sulfur_dioxide, tf.float32)
        volatile_acidity_tensor = tf.dtypes.cast(volatile_acidity, tf.float32)

        parsed_features = {'alcohol': alcohol_tensor,
                           'chlorides': chlorides_tensor,
                           'citric_acid': citric_acid_tensor,
                           'density': density_tensor,
                           'fixed_acidity': fixed_acidity_tensor,
                           'free_sulfur_dioxide': free_sulfur_dioxide_tensor,
                           'ph': ph_tensor,
                           'residual_sugar': residual_sugar_tensor,
                           'sulphates': sulphates_tensor,
                           'total_sulfur_dioxide': total_sulfur_dioxide_tensor,
                           'volatile_acidity': volatile_acidity_tensor,
                           }

        transformed_features = model.tft_layer(parsed_features)

        return model(transformed_features)

    return serve_raw_fn


def _make_keras_model() -> tf.keras.Model:
    """Creates a Keras model for classifying wine quality.

    Returns:
      A Keras Model.
    """
    inputs = [keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]

    model_layers = keras.layers.concatenate(inputs)
    model_layers = keras.layers.Dense(units=20, activation='relu')(model_layers)
    model_layers = keras.layers.Dropout(0.1)(model_layers)
    model_layers = keras.layers.BatchNormalization()(model_layers)
    model_layers = keras.layers.Dense(units=8, activation='relu')(model_layers)
    model_layers = keras.layers.Dropout(0.1)(model_layers)
    model_layers = keras.layers.BatchNormalization()(model_layers)
    model_layers = keras.layers.Dense(units=12, activation='relu')(model_layers)
    model_layers = keras.layers.Dropout(0.1)(model_layers)
    model_layers = keras.layers.BatchNormalization()(model_layers)

    outputs = keras.layers.Dense(1, activation='sigmoid')(model_layers)

    model = keras.Model(inputs=inputs, outputs=outputs)
    rms_prop_optimizer = tf.keras.optimizers.RMSprop(learning_rate=10e-3)
    model.compile(
        optimizer=rms_prop_optimizer,
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['accuracy']
    )

    model.summary(print_fn=logging.info)
    return model


def run_fn(fn_args: tfx.components.FnArgs):
    """Train the model based on given args.

    Args:
      fn_args: Holds args used to train the model as name/value pairs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        tf_transform_output,
        50,
        batch_size=_TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(
        fn_args.eval_files,
        tf_transform_output,
        50,
        batch_size=_EVAL_BATCH_SIZE)

    model = _make_keras_model()
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=_NB_EPOCH)

    # defines serving signatures : default and raw data
    signatures = {
        'serving_default':
            _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
                tf.TensorSpec(
                    shape=[None],
                    dtype=tf.string,
                    name='examples')),
        "serving_raw": _get_serve_raw(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='alcohol'),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='chlorides'),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='citric_acid'),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='density'),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='fixed_acidity'),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='free_sulfur_dioxide'),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='ph'),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='residual_sugar'),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='sulphates'),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='total_sulfur_dioxide'),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='volatile_acidity'),
        )
    }

    # The result of the training should be saved in `fn_args.serving_model_dir`
    # directory.
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
