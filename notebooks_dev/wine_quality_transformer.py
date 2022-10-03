
import tensorflow as tf
import tensorflow_transform as tft

_NUMERIC_FEATURE_KEYS = [
    'fixed_acidity',	'volatile_acidity',	'citric_acid',	'residual_sugar',
    'chlorides',	'free_sulfur_dioxide',	'total_sulfur_dioxide',	'density',
    'ph',	'sulphates',	'alcohol'
]
_LABEL_KEY = 'quality'

def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
        inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
        Map from string feature key to transformed feature operations.
    """

    outputs = {}

    # scale features to [0,1]
    for key in _NUMERIC_FEATURE_KEYS:
        scaled = tft.scale_to_0_1(inputs[key])
        outputs[key] = tf.reshape(scaled, [-1])

    # transform the output
    outputs[_LABEL_KEY] = inputs[_LABEL_KEY]

    return outputs
