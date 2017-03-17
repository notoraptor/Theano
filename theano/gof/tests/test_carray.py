from __future__ import print_function, absolute_import, division
from theano.gof import Op, Apply, CArrayType
import theano.scalar as scalar
import theano


class FracSum(Op):
    params_type = CArrayType('double')

    def __init__(self, fractions):
        assert isinstance(fractions, (list, tuple))
        self.fractions = tuple(fractions)

    def get_params(self, node):
        return self.fractions

    def make_node(self, base):
        return Apply(self, [scalar.as_scalar(base)], [scalar.float64()])

    def perform(self, node, inputs, output_storage, fractions):
        base, = inputs
        result, = output_storage
        result[0] = sum(base / denominator for denominator in fractions)

    def c_code(self, node, name, inputs, outputs, sub):
        return """
        %(y)s = 0;
        for (int i = 0; i < %(fractions)s->length; ++i) {
            %(y)s += %(a)s/%(fractions)s->array[i];
        }
        """ % dict(y=outputs[0], fractions=sub['params'], a=inputs[0])


def test_carray():
    fractions = (1, 2, 3, 4, 5)
    a = scalar.int32()
    y = FracSum(fractions)(a)
    f = theano.function([a], y)
    refs = tuple((sum(base / denominator for denominator in fractions)) for base in (1.0, 2.0, 3.0, 120.0))
    outs = (f(1), f(2), f(3), f(120))
    assert refs == outs, (refs, outs)
