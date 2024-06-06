import torch

def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    """
    For every value v in `labels`, the value in the output will be either 1 or 0. Each vector along the `dim`-th
    dimension has the "one-hot" format, i.e., it has a total length of `num_classes`,
    with a one and `num_class-1` zeros.
    Note that this will include the background label, thus a binary mask should be treated as having two classes.

    Args:
        labels: input tensor of integers to be converted into the 'one-hot' format. Internally `labels` will be
            converted into integers `labels.long()`.
        num_classes: number of output channels, the corresponding length of `labels[dim]` will be converted to
            `num_classes` from `1`.
        dtype: the data type of the output one_hot label.
        dim: the dimension to be converted to `num_classes` channels from `1` channel, should be non-negative number.

    Example:

    For a tensor `labels` of dimensions [B]1[spatial_dims], return a tensor of dimensions `[B]N[spatial_dims]`
    when `num_classes=N` number of classes and `dim=1`.

    .. code-block:: python

        from monai.networks.utils import one_hot
        import torch

        a = torch.randint(0, 2, size=(1, 2, 2, 2))
        out = one_hot(a, num_classes=2, dim=0)
        print(out.shape)  # torch.Size([2, 2, 2, 2])

        a = torch.randint(0, 2, size=(2, 1, 2, 2, 2))
        out = one_hot(a, num_classes=2, dim=1)
        print(out.shape)  # torch.Size([2, 2, 2, 2, 2])

    """

    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels


def one_hot_reverse(predicted, dim):
    return torch.argmax(predicted, dim=dim)