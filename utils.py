import json


def decode_predictions(preds, top=5):
    """Decodes the prediction of an ImageNet model.

    Arguments:
      preds: Numpy array encoding a batch of predictions.
      top: Integer, how many top-guesses to return. Defaults to 5.

    Returns:
      A list of lists of top class prediction tuples
      `(class_name, class_description, score)`.
      One list of tuples per sample in batch input.

    Raises:
      ValueError: In case of invalid shape of the `pred` array
        (must be 2D).
    """
    global CLASS_INDEX

    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    # if CLASS_INDEX is None:
    #     fpath = data_utils.get_file(
    #         'imagenet_class_index.json',
    #         CLASS_INDEX_PATH,
    #         cache_subdir='models',
    #         file_hash='c2c37ea517e94d9795004a39431a14cb')
    #     with open(fpath) as f:
    #         CLASS_INDEX = json.load(f)
    CLASS_INDEX = json.load(open("./model/imagenet_class_index.json"))

    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results
