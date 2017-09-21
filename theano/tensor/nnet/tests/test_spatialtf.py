from __future__ import absolute_import, print_function, division

import numpy as np
import theano
import theano.ifelse
from theano import tensor as T
from theano import config
from theano.tests import unittest_tools as utt
from theano.tensor.nnet.spatialtf import spatialtf, TransformerGrid, TransformerSampler, TransformerGradI, \
    TransformerGradT
from theano.tensor.nnet.symbolic_spatialtf import theano_spatialtf


def assert_raises(exception_classes, callable, *args):
    try:
        callable(*args)
    except Exception as e:
        if not isinstance(e, tuple(exception_classes)):
            raise e
    else:
        raise AssertionError('assert_raises', exception_classes, callable.__name__, *args)


class TestTransformer(object):
    mode = None
    transformer_grid_op = TransformerGrid
    transformer_sampler_op = TransformerSampler
    transformer_grad_i_op = TransformerGradI
    transformer_grad_t_op = TransformerGradT

    some_transformations = [
        [[1, 0, 0],
         [0, 1, 0]],
        [[-1, 0, 0],
         [0, -1, 0]],
        [[1, 0, -1],
         [0, 1, -1]],
        [[1, 1, 1],
         [1, 1, 1]],
        [[-1.90120868, 1.48872078, -4.01530816],
         [8.27449531, 1.75634363, 6.66776181]]
    ]
    symb_out_fn = None
    impl_out_fn = None
    inp_cases = []
    transform_cases = []

    def __init__(self):
        utt.seed_rng()
        t_inp = T.tensor4('inp')
        t_theta = T.tensor3('theta')
        t_scale_height = T.scalar('scale_height')
        t_scale_width = T.scalar('scalar_width')

        symb_out_var = theano_spatialtf(t_inp, t_theta, t_scale_width, t_scale_height)
        cpu_out_var = spatialtf(t_inp, t_theta, t_scale_width, t_scale_height)
        self.symb_out_fn = theano.function([t_inp, t_theta, t_scale_width, t_scale_height], symb_out_var,
                                           mode=self.mode)
        self.impl_out_fn = theano.function([t_inp, t_theta, t_scale_width, t_scale_height], cpu_out_var, mode=self.mode)

        count_grid_ops = 0
        count_sampler_ops = 0
        for node in self.impl_out_fn.maker.fgraph.apply_nodes:
            if isinstance(node.op, self.transformer_grid_op):
                count_grid_ops += 1
            elif isinstance(node.op, self.transformer_sampler_op):
                count_sampler_ops += 1
        assert count_grid_ops == 1, count_grid_ops
        assert count_sampler_ops == 1, count_sampler_ops

        cases_count = 10
        self.inp_cases = self._get_inp_shape_cases(cases_count)
        self.transform_cases = self._get_transform_cases(cases_count - len(self.some_transformations))

    def setUp(self):
        utt.seed_rng()

    def _get_inp_shape_cases(self, count):
        return [self._generate_inp_shape() for i in range(count)]

    def _get_transform_cases(self, count):
        return [np.asarray(t) for t in self.some_transformations] + [self._generate_transformation() for i in
                                                                     range(count)]

    def _generate_inp_shape(self):
        return (
            np.random.randint(1, 11), np.random.randint(1, 6), np.random.randint(10, 101), np.random.randint(10, 101))

    def _generate_transformation(self):
        return np.random.uniform(-10, 10, (2, 3)).astype(theano.config.floatX)

    def getInputs(self, inp_shape, transform):
        inp = np.random.random(inp_shape).astype(config.floatX)
        theta = np.asarray(inp_shape[0] * [transform], dtype=config.floatX)
        scale_height = np.random.random()
        scale_width = np.random.random()
        return (inp, theta, scale_width, scale_height)

    def compare_implementations_numpy_theano(self, case_index):
        # Compare CPU implementation with symbolic implementation.
        inp_shape = self.inp_cases[case_index]
        transform = self.transform_cases[case_index]
        inp, theta, scale_width, scale_height = self.getInputs(inp_shape, transform)

        symb_out = self.symb_out_fn(inp, theta, scale_width, scale_height)
        cpu_out = self.impl_out_fn(inp, theta, scale_width, scale_height)

        rtol = None
        atol = rtol

        try:
            utt.assert_allclose(cpu_out, symb_out, atol=atol, rtol=rtol)
        except Exception as e:
            print('Failing case:')
            print('Input shape:', inp_shape)
            print('Transform:')
            print(transform)
            raise e

    def test_symbolic_implementation(self):
        for test_case_index in range(len(self.inp_cases)):
            yield (self.compare_implementations_numpy_theano, test_case_index)

    def test_grad_input(self):
        inp, theta, scale_width, scale_height = self.getInputs((3, 3, 5, 5), self._generate_transformation())

        def grad_inp_functor(inputs):
            out = spatialtf(inputs, theta, scale_width, scale_height)
            return out

        utt.verify_grad(grad_inp_functor, [inp], n_tests=10, mode=self.mode)

    def test_grad_theta(self):
        inp, theta_val, scale_width, scale_height = self.getInputs((3, 3, 5, 5), self._generate_transformation())

        def grad_theta_functor(theta):
            out = spatialtf(inp, theta, scale_width, scale_height)
            return out

        utt.verify_grad(grad_theta_functor, [theta_val], n_tests=10, mode=self.mode)

    def test_grad_grid(self):
        inp, theta_val, scale_width, scale_height = self.getInputs((3, 3, 5, 5), self._generate_transformation())
        out_dims = [int(v) for v in (inp.shape[0], inp.shape[1],
                                     np.ceil(inp.shape[2] * scale_height),
                                     np.ceil(inp.shape[3] * scale_width))]
        inp = theano.shared(inp)
        theta_val = theano.shared(theta_val)
        grid_var = self.transformer_grid_op()(theta_val, out_dims)
        fn_grid = theano.function([], grid_var, mode=self.mode)
        grid_val = fn_grid()

        def grad_grid_functor(grid):
            out = self.transformer_sampler_op()(inp, grid)
            return out

        utt.verify_grad(grad_grid_functor, [grid_val], n_tests=10, mode=self.mode)

    def test_grad(self):
        utt.seed_rng()

        inputs = T.tensor4('inputs')
        theta = T.tensor3('theta')

        out = spatialtf(inputs, theta, scale_height=0.25, scale_width=0.75)
        out_mean = T.mean(out)
        mean_gi = T.grad(out_mean, [inputs])
        mean_gt = T.grad(out_mean, [theta])

        f_gi = theano.function([inputs, theta], mean_gi, mode=self.mode)
        assert any([isinstance(node.op, self.transformer_grad_i_op)
                    for node in f_gi.maker.fgraph.apply_nodes])

        f_gt = theano.function([inputs, theta], mean_gt, mode=self.mode)
        assert any([isinstance(node.op, self.transformer_grad_t_op)
                    for node in f_gt.maker.fgraph.apply_nodes])

        input_dims = (5, 3, 16, 16)
        inputs_val = np.random.random(size=input_dims).astype(theano.config.floatX)
        inputs_val = np.ones(input_dims)

        # Tensor with transformations
        theta_val = np.random.random((input_dims[0], 2, 3)).astype(theano.config.floatX)
        theta_val = np.ones((input_dims[0], 2, 3))
        # Using smaller values for theta, increases the precision of gradients
        # when using lower precision. Tests might fail for lower precision data
        # types if the values of theta or the inputs are very high.
        # theta /= 100

        # Check that the gradients are computed
        f_gi(inputs_val, theta_val)
        f_gt(inputs_val, theta_val)

        def grad_functor(inputs, theta):
            out = spatialtf(inputs, theta)
            return out

        atol, rtol = None, None
        # if theano.config.floatX == 'float32':
        #     rtol = 5e-2
        # elif theano.config.floatX == 'float16':
        #     rtol = 1e-0

        # TODO: currently fails, even in float64.
        utt.verify_grad(grad_functor, [inputs_val, theta_val], abs_tol=atol, rel_tol=rtol, mode=self.mode)

    def test_invalid_shapes(self):
        inputs = T.tensor4('inputs')
        theta = T.tensor3('theta')

        st = spatialtf(inputs, theta)
        st_func = theano.function([inputs, theta], st, mode=self.mode)

        inputs_val = np.ones((3, 5, 7, 7), dtype=theano.config.floatX)

        def try_theta_shp(theta_shp):
            theta_val = np.ones(theta_shp, dtype=theano.config.floatX)
            return st_func(inputs_val, theta_val)

        # the theta shape for this input should be (3, 2, 3)
        try_theta_shp((3, 2, 3))

        # incorrect parameter dimensions
        assert_raises([ValueError, RuntimeError], try_theta_shp, (3, 1, 3))
        assert_raises([ValueError, RuntimeError], try_theta_shp, (3, 2, 1))

        # number of rows does not match the number of input rows
        assert_raises([ValueError, RuntimeError], try_theta_shp, (1, 2, 3))
        assert_raises([ValueError, RuntimeError], try_theta_shp, (4, 2, 3))
