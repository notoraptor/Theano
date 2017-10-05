"""
Concrete implementation of spatial transformer
( by Jaderberg et. al: http://arxiv.org/abs/1506.02025 )
"""
from __future__ import absolute_import, print_function, division

import theano
from theano.gof.op import Op
from theano.scalar import as_scalar
from theano.tensor import as_tensor_variable
from theano.tensor.extra_ops import cpu_contiguous
from theano.gof import Apply
from theano.gradient import grad_not_implemented
from itertools import product
import numpy as np

numpy_version = tuple(int(x) for x in np.version.short_version.split('.'))
if numpy_version >= (1, 10, 0):
    matmul = np.matmul
else:
    def matmul(a, b):
        # To support older NumPy versions.
        # May be better implemented.
        assert a.ndim == b.ndim >= 2
        if a.ndim == 2:
            return np.dot(a, b)
        common_dtype = np.find_common_type([a.dtype, b.dtype], [])
        out = np.empty(a.shape[:-1] + b.shape[-1:], common_dtype)
        for i in range(a.ndim - 2):
            assert a.shape[i] == b.shape[i] or b.shape[i] == 1
        assert a.shape[-1] == b.shape[-2]
        coordinates_a = []
        coordinates_b = []
        count_positions = 1
        for i in range(a.ndim - 2):
            coordinates = list(range(a.shape[i]))
            coordinates_a += [coordinates]
            if b.shape[i] == 1:
                coordinates_b += [a.shape[i] * [0]]
            else:
                coordinates_b += [coordinates]
            count_positions *= a.shape[i]
        positions_a = product(*coordinates_a)
        positions_b = product(*coordinates_b)
        for i in range(count_positions):
            a_pos = next(positions_a)
            b_pos = next(positions_b)
            out[a_pos] = np.dot(a[a_pos], b[b_pos])
        return out


required_dtypes = ('float16', 'float32', 'float64')


def sampling_grid(height, width, dtype):
    """
    Create sampling grid.
    """
    x_t, y_t = np.meshgrid(np.linspace(-1, 1, width, dtype=dtype),
                           np.linspace(-1, 1, height, dtype=dtype))
    ones = np.ones(np.prod(x_t.shape), dtype=dtype)
    grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    return grid


def scale(v, limit):
    """
    Scale value v from frame [-1; 1] to frame [0; limit - 1].
    (v may be not in interval [-1; 1]).
    """
    return ((v + 1) * (limit - 1)) / 2


def k(values):
    """
    Bilinear sampling kernel.
    """
    return np.maximum(np.zeros(values.shape, values.dtype), 1 - np.abs(values))


class TransformerGrid(Op):
    """
    Grid generator Op for a spatial transformer.
    """
    __props__ = ()

    def make_node(self, theta, out_dims):
        """
        Create a grid generator node for a spatial transformer

        Parameters
        ----------
        theta : tensor
            Affine transformation tensor containing one affine transformation
            matrix per image. ``theta`` is usually generated by the localization
            network.

        out_dims : tuple
            Dimensions of the transformed inputs, containing four elements, and is given
            by (N, C, H, W), where N is the number of inputs, C the number of channels,
            H and W are the height and width of each input.
        """
        theta = cpu_contiguous(as_tensor_variable(theta))

        if theta.ndim != 3:
            raise TypeError('TransformerGrid (make_node) requires theta to '
                            'be a 3D tensor; received "%s" (%i dims)' %
                            (theta, theta.ndim))

        assert theta.type.dtype in ('float16', 'float32', 'float64')

        out_dims = cpu_contiguous(as_tensor_variable(out_dims))

        if out_dims.ndim != 1:
            raise TypeError('TransformerGrid requires a vector for out_dims, got %d dimensions' %
                            out_dims.ndim)

        if out_dims.dtype not in theano.tensor.basic.integer_dtypes:
            raise TypeError('TransformerGrid requires integers as out_dims, got dtype %s' %
                            out_dims.dtype)

        out_dims = theano.tensor.basic.cast(out_dims, 'int64')

        grid = theano.tensor.tensor(dtype=theta.dtype,
                                    broadcastable=(theta.ndim + 1) * (False,))

        return Apply(self, [theta, out_dims], [grid])

    def perform(self, node, inputs, output_storage):
        theta, out_dims = inputs

        # Only 2D images are currently supported.
        if len(out_dims) != 4:
            raise NotImplementedError('SpatialTransformerGrid currently supports only 2D data (4D tensors).')

        # Theta should be in the format (batch_size, 2, 3)
        if tuple(theta.shape) != (out_dims[0], 2, 3):
            raise ValueError('TransformerGrid: theta must have shape (N, 2, 3) '
                             'where N is the batch size, got (%s, %s, %s)' %
                             (theta.shape[0], theta.shape[1], theta.shape[2]))

        num_batch = theta.shape[0]
        grid_out = output_storage[0]

        out_height, out_width = out_dims[2:]
        grid = sampling_grid(out_height, out_width, theta.dtype)
        # Generate transformed grid with shape (num_batch, 2, out_height * out_width)
        transformed_grid = np.dot(theta, grid)
        # Reshape into (num_batch, out_height, out_width, 2)
        transposed_grid = transformed_grid.reshape(num_batch, 2, out_height, out_width).transpose(0, 2, 3, 1)
        grid_out[0] = transposed_grid.astype(theta.dtype)

    def grad(self, inputs, grads):
        theta, out_dims = inputs
        dgrid = grads[0]

        dtheta = TransformerGradT()(theta, dgrid)
        return [dtheta, grad_not_implemented(self, 1, out_dims)]


class TransformerSampler(Op):
    """
    Grid sampler Op for a spatial transformer.
    """
    __props__ = ()

    def make_node(self, inp, grid):
        """
        Create a grid sampler node for a spatial transformer

        Parameters
        ----------
        inp : tensor
            Images from which the pixels will be sampled. The implementation
            assumes the tensor is in NCHW format, where N is the number of images,
            C is the number of color channels, H is the height of the inputs, and
            W is width of the inputs.

        grid : TransformerGrid
            Grid that contains the coordinates of the pixels to be sampled from
            the input images.
        """
        inp = cpu_contiguous(as_tensor_variable(inp))

        if inp.ndim != 4:
            raise TypeError('TransformerSampler (make_node) requires input to '
                            'be a 4D tensor; received "%s" (%i dims)' %
                            (inp, inp.ndim))

        assert inp.dtype in ('float16', 'float32', 'float64')

        grid = cpu_contiguous(as_tensor_variable(grid))

        if grid.ndim != 4:
            raise TypeError('TransformerSampler (make_node) requires grid to '
                            'be a 4D tensor; received "%s" (%i dims)' %
                            (grid, grid.ndim))

        assert grid.dtype in ('float16', 'float32', 'float64')

        out = inp.type()

        return Apply(self, [inp, grid], [out])

    def grad(self, inputs, grads):
        inp, grid = inputs
        grad_outputs = grads[0]

        grad_inp, grad_grid = TransformerGradI()(inp, grid, grad_outputs)

        return [grad_inp, grad_grid]

    def perform(self, node, inputs, outputs_storage):
        inp, grid = inputs
        out_height, out_width = grid.shape[1], grid.shape[2]
        num_batch, num_channels, N, M = inp.shape
        assert num_batch == grid.shape[0]
        assert grid.shape[3] == 2
        # Q is the number of output points.
        Q = out_height * out_width

        # num_batch, Q, 2
        grid_reshaped = grid.reshape(num_batch, out_height * out_width, 2)
        # num_batch, Q, 1
        all_x, all_y = np.split(grid_reshaped, 2, axis=2)
        # scale x wrt/ M
        all_x = scale(all_x, M)
        # scale y wrt/ N
        all_y = scale(all_y, N)
        # num_batch, Q, 2 for x and y ( [x;-1]..., [y;-1]... )
        all_neg_ones = -np.ones((num_batch, Q, 1), grid.dtype)
        all_x = np.concatenate((all_x, all_neg_ones), axis=2)
        all_y = np.concatenate((all_y, all_neg_ones), axis=2)
        # 2, M ( [1, ...][0, 1, ... M-1] )
        M1 = np.vstack((np.ones(M, grid.dtype), np.arange(M, dtype=grid.dtype)))
        # 2, N ( [1...][0, 1, ... N-1] )
        N1 = np.vstack((np.ones(N, grid.dtype), np.arange(N, dtype=grid.dtype)))
        # num_batch, Q, M, then num_batch, Q, 1, M ( all x - m for every x in grid and every m in [0, m) )
        all_kxm = k(np.dot(all_x, M1)).reshape(num_batch, Q, 1, M)
        # num_batch, Q, N, then num_batch, Q, N, 1 ( all y -n for every y in grid and every n in [0, n) )
        all_kyn = k(np.dot(all_y, N1)).reshape(num_batch, Q, N, 1)
        # num_batch, Q, N, M ( all (x - m)(y - n) )
        all_kyx = matmul(all_kyn, all_kxm)
        # num_batch, Q, N*M
        b_q_nm = all_kyx.reshape(num_batch, Q, N * M)
        # num_batch, N*M, C
        b_nm_c = inp.reshape(num_batch, num_channels, N * M).transpose(0, 2, 1)
        # num_batch, Q, C ( sum{ img[n, m] * (x - m) * (y - n) } in every channel for each output point (x, y) )
        b_q_c = matmul(b_q_nm, b_nm_c)
        # num_batch, num_channels, out_height, out_width
        out = b_q_c.transpose(0, 2, 1).reshape(num_batch, num_channels, out_height, out_width)

        outputs_storage[0][0] = out


class TransformerGradI(Op):
    """
    Gradient of inputs Op for a spatial transformer.
    """
    __props__ = ()

    def make_node(self, inp, grid, grad_outputs):
        """
        Create a gradient of the inputs' node for a spatial transformer

        Parameters
        ----------
        inp : tensor
            Images from which the pixels will be sampled. The implementation
            assumes the tensor is in NCHW format, where N is the number of images,
            C is the number of color channels, H is the height of the inputs, and
            W is width of the inputs.

        grad_outputs : tensor
            Gradients of the sampled outputs.
        """
        inp = cpu_contiguous(as_tensor_variable(inp))
        assert inp.ndim == 4
        assert inp.dtype in required_dtypes

        grid = cpu_contiguous(as_tensor_variable(grid))
        assert grid.ndim == 4
        assert grid.dtype in required_dtypes

        grad_outputs = cpu_contiguous(as_tensor_variable(grad_outputs))
        assert grad_outputs.ndim == 4
        assert grad_outputs.dtype in required_dtypes

        grad_inp = inp.type()

        grad_grid = grid.type()

        return Apply(self, [inp, grid, grad_outputs], [grad_inp, grad_grid])

    def perform(self, node, inputs, output_storage):
        inp, grid, grad_outputs = inputs
        grad_inp_out = output_storage[0]
        grad_grid_out = output_storage[1]

        out_height, out_width = grid.shape[1], grid.shape[2]
        num_batch, num_channels, N, M = inp.shape
        assert num_batch == grid.shape[0]
        assert grid.shape[3] == 2

        # Q is the number of output points.
        Q = out_height * out_width
        # num_batch, Q, 2
        grid_reshaped = grid.reshape(num_batch, out_height * out_width, 2)
        # num_batch, Q, 1
        all_x, all_y = np.split(grid_reshaped, 2, axis=2)
        # scale x wrt/ M
        all_x = scale(all_x, M)
        # scale y wrt/ N
        all_y = scale(all_y, N)
        # num_batch, Q, 2 for x and y ( [x;-1]..., [y;-1]... )
        all_neg_ones = -np.ones((num_batch, Q, 1), grid.dtype)
        all_x = np.concatenate((all_x, all_neg_ones), axis=2)
        all_y = np.concatenate((all_y, all_neg_ones), axis=2)
        # 2, M ( [1, ...][0, 1, ... M-1] )
        M1 = np.vstack((np.ones(M, grid.dtype), np.arange(M, dtype=grid.dtype)))
        # 2, N ( [1...][0, 1, ... N-1] )
        N1 = np.vstack((np.ones(N, grid.dtype), np.arange(N, dtype=grid.dtype)))
        # num_batch, Q, M, then num_batch, Q, 1, M ( all x - m for every x in grid and every m in [0, m) )
        all_kxm = k(np.dot(all_x, M1)).reshape(num_batch, Q, 1, M)
        # num_batch, Q, N, then num_batch, Q, N, 1 ( all y -n for every y in grid and every n in [0, n) )
        all_kyn = k(np.dot(all_y, N1)).reshape(num_batch, Q, N, 1)
        # num_batch, Q, N, M ( all (x - m)(y - n) )
        all_kyx = matmul(all_kyn, all_kxm)

        # Let's compute grad input
        b_nm_q = all_kyx.reshape(num_batch, Q, N * M).transpose(0, 2, 1)
        b_q_c = grad_outputs.reshape(num_batch, num_channels, Q).transpose(0, 2, 1)
        b_nm_c = matmul(b_nm_q, b_q_c)
        grad_inp_out[0] = b_nm_c.transpose(0, 2, 1).reshape(num_batch, num_channels, N, M)

        # Let's compute grad grid (currently failing ...)

        def gradient_k(val):
            # Should compute gradient of bilinear sampling.
            v1_pos = np.logical_and(val <= 0, val > -1).astype(np.int8)
            v1_neg = -np.logical_and(val > 0, val < 1).astype(np.int8)
            v_grad = np.add(v1_pos, v1_neg)
            # Note: we will have 0 everywhere |val| >= 1.
            return v_grad.astype(val.dtype)

        # num_batch, Q, 1, M
        x_less_m = np.dot(all_x, M1)
        xm_for_grad_x = gradient_k(x_less_m).reshape(num_batch, Q, 1, M)
        # num_batch, Q, N, 1
        y_less_n = np.dot(all_y, N1)
        yn_for_grad_y = gradient_k(y_less_n).reshape(num_batch, Q, N, 1)
        # grad grid
        kyx_for_grad_x = matmul(all_kyn, xm_for_grad_x).reshape(num_batch, Q, 1, N * M)
        kyx_for_grad_y = matmul(yn_for_grad_y, all_kxm).reshape(num_batch, Q, 1, N * M)
        # num_batch, Q, 2, N * M
        kyx_for_grad = np.concatenate((kyx_for_grad_x, kyx_for_grad_y), axis=2)
        # num_batch, 1, N*M, C
        img_reshaped = inp.reshape(num_batch, 1, num_channels, N * M).transpose(0, 1, 3, 2)
        # num_batch, Q, 2, C (is the error here ?)
        grad_grid = matmul(kyx_for_grad, img_reshaped)
        b_q_c_1 = grad_outputs.reshape(num_batch, num_channels, Q).transpose(0, 2, 1).reshape(num_batch, Q,
                                                                                              num_channels, 1)
        # num_batch, Q, 2, 1 => num_batch, H, W, 2
        grad_grid_weighted = matmul(grad_grid, b_q_c_1).reshape(num_batch, out_height, out_width, 2)
        grad_grid_out[0] = grad_grid_weighted


class TransformerGradT(Op):
    """
    Gradient of affine transformations Op for a spatial transformer.
    """
    __props__ = ()

    def make_node(self, theta, grad_grid):
        """
        Create a gradient of the transform node for a spatial transformer

        Parameters
        ----------
        theta : tensor
            Affine transformation tensor containing one affine transformation
            matrix per image. ``theta`` is usually generated by the localization
            network.

        grad_grid : tensor
            Gradients of the sampling grid.
        """
        theta = as_tensor_variable(theta)
        assert theta.ndim == 3
        assert theta.dtype in required_dtypes

        grad_grid = as_tensor_variable(grad_grid)
        assert grad_grid.ndim == 4
        assert grad_grid.dtype in required_dtypes

        out = theano.tensor.tensor(dtype=theta.type.dtype,
                                   broadcastable=theta.broadcastable)

        return Apply(self, [theta, grad_grid], [out])

    def perform(self, node, inputs, output_storage):
        theta, grad_grid = inputs
        out = output_storage[0]

        num_batch = theta.shape[0]
        assert num_batch == grad_grid.shape[0]

        out_height, out_width = grad_grid.shape[1], grad_grid.shape[2]
        assert grad_grid.shape[3] == 2

        grid = sampling_grid(out_height, out_width, theta.dtype)
        # (3, h * w) -> (h * w, 3)
        transposed_grid = grid.transpose(1, 0)
        # reshape gradients of grid from (n, h, w, 2) -> (n, h * w, 2) -> (n, 2, h * w)
        _grad_grid_transposed = grad_grid.reshape(num_batch, out_height * out_width, 2).transpose(0, 2, 1)
        out[0] = np.dot(_grad_grid_transposed, transposed_grid)


def spatialtf(img, theta, scale_width=1, scale_height=1):
    """
    Spatial transformer (by Jaderberg et. al).

    Parameters
    ----------
    img : tensor
        Images to which the transformations will be applied. The implementation
        assumes the tensor is in NCHW format, where N is the number of images,
        C is the number of color channels, H is the height of the inputs, and
        W is width of the inputs.
    theta : tensor
        Affine transformation tensor containing one affine transformation
        matrix per image. ``theta`` is usually generated by the localization
        network.
    scale_height: float
        A float specifying the scaling factor for the height of the output
        image. A value of 1 will keep the original height of the input. Values
        larger than 1 will upsample the input. Values below 1 will downsample
        the input.
    scale_width: float
        A float specifying the scaling factor for the width of the output
        image. A value of 1 will keep the original width of the input. Values
        larger than 1 will upsample the input. Values below 1 will downsample
        the input.

    Returns
    -------
    out : tensor
        Transformed images with width and height properly scaled.

    Notes
    -----
    Currently, only 2D transformations with 2x3 affine transformation matrices
    are supported.
    """
    img = as_tensor_variable(img)

    theta = as_tensor_variable(theta)

    num_batch, num_channels, height, width = img.shape
    out_height = theano.tensor.cast(theano.tensor.ceil(scale_height * height), 'int64')
    out_width = theano.tensor.cast(theano.tensor.ceil(scale_width * width), 'int64')

    out_dims = (num_batch, num_channels, out_height, out_width)
    out_dims = tuple([as_scalar(v).astype('int64') for v in out_dims])

    grid = TransformerGrid()(theta, out_dims)
    sampler = TransformerSampler()(img, grid)
    return sampler
