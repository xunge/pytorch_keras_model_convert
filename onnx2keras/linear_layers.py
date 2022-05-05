from tensorflow import keras
import logging
from .utils import is_numpy

def convert_gemm(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Linear / GEMM layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.gemm')

    # Check if Bias available
    if len(node.input) == 3:
        has_bias = True
        keras_weights = [layers[node.input[1]], layers[node.input[2]]]
        logger.debug('Convert GEMM with bias.')
    elif len(node.input) == 2:
        has_bias = False
        keras_weights = [layers[node.input[1]]]
        logger.debug('Convert GEMM without bias.')
    else:
        raise AttributeError('More than 3 or less than 2 inputs')

    # Linear can have additional flag to transpose weights
    if 'transB' in params and params['transB'] == 1:
        logger.debug('Transposing W matrix.')
        keras_weights[0] = keras_weights[0].transpose()

    # Estimate input/output neurons
    input_channels, output_channels = keras_weights[0].shape
    logger.debug('Input units %s, output units %s.', input_channels, output_channels)

    if is_numpy(keras_weights[0]):
        dense = keras.layers.Dense(
            output_channels,
            weights=keras_weights, name=keras_name, bias_initializer='zeros', kernel_initializer='zeros', use_bias=has_bias
        )

        # The first input - always X
        try:
            layers[node_name] = dense(layers[node.input[0]])
        except ValueError:
            reshape = keras.layers.Reshape([input_channels], name=keras_name + '_reshape')
            reshaped_x = reshape(layers[node.input[0]])
            layers[node_name] = dense(reshaped_x)
    
    else:
        layers[node_name] = keras.layers.Multiply()([layers[node.input[0]], layers[node.input[1]]])
