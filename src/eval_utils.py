
def top5UptoD_err(y_true, y_pred):
    """ Evaluation metric used by rclc.
        This metric prioritizes precision and take into account a variable
        number of datasets per publication.
        Reference: https://github.com/Coleridge-Initiative/rclc#evaluation
        Input:
            - y_true: ground truths
            - y_pred: top 5 predictions
    """
    if len(y_true) == 0:
        raise ValueError(f'Error: No ground truth!')
    if len(y_pred) != 5:
        raise ValueError(f'the length of y_pred must equal to 5.')

    _D = len(y_true) if len(y_true) <= 5 else 5
    correct = 0
    error = 0
    m = 0
    for di in y_pred:
        if di in y_true:
            correct += 1
        else:
            error += 1
        m += 1
        if correct == _D:
            break
    return (error * 1.) / m


def top5UptoD_prec(y_true, y_pred):
    """ Evaluation metric used by rclc.
        This metric prioritizes precision and take into account a variable
        number of datasets per publication.
        Reference: https://github.com/Coleridge-Initiative/rclc#evaluation
        Input:
            - y_true: ground truths
            - y_pred: top 5 predictions
    """
    if len(y_true) == 0:
        raise ValueError(f'Error: No ground truth!')
    if len(y_pred) != 5:
        raise ValueError(f'the length of y_pred must equal to 5.')

    _D = len(y_true) if len(y_true) <= 5 else 5
    correct = 0
    error = 0
    m = 0
    for di in y_pred:
        if di in y_true:
            correct += 1
        else:
            error += 1
        m += 1
        if correct == _D:
            break
    return (correct * 1.) / m
