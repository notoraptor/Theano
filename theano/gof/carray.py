from __future__ import print_function, absolute_import, division
import re
import theano
from theano.gof import Type, Wrapper


class CArrayFromTuple(Type):
    """"
    Op parameter internal class to handle a Python vector of primitive types (integer, floating values or booleans).
    Booleans are converted to integers (False => 0, True -> 1).
    Vector must be represented as a tuple of values.

    In C, vectors are converted to a dynamically allocated C array,
    which C type must be specified when creating the param type::

        carray_from_tuple = CArrayFromTuple('int')
        good_value = (1, 2, 3, 4, 5)

    In C code:

    .. code-block:: c

        int* good_value;
        /*
        good_value will be allocated and filled from its Python's `good_value` counterpart.
        ...
        */
        int v = good_value[1]; // v == 2

    .. warning::

        This class handles only creation and destruction of an array in C,
        thus it does not provide any way to know array length into C code.
        You should better use :class:`CArrayType` which provides both
        array data and array length in C code.

    """

    def check_ctype(self):
        els = self.ctype.split()
        if not all(re.match('^[A-Za-z_][A-Za-z0-9_]*$', el) for el in els):
            raise TypeError('%s: invalid C type "%s".' % (type(self).__name__, self.ctype))
        self.ctype = ' '.join(els)

    def __init__(self, ctype):
        self.ctype = ctype
        self.check_ctype()

    def __repr__(self):
        return '%s<%s>' % (type(self).__name__, self.ctype)

    def __eq__(self, other):
        return type(self) == type(other) and self.ctype == other.ctype

    def __hash__(self):
        return hash((type(self), self.ctype))

    def filter(self, data, strict=False, allow_downcast=None):
        if strict:
            assert isinstance(data, tuple) and all(isinstance(value, (int, float)) for value in data)
            return data
        assert isinstance(data, (list, tuple))
        assert all(isinstance(value, (int, float, bool)) for value in data)
        data = tuple(int(value) if isinstance(value, bool) else value for value in data)
        return data

    def values_eq(self, a, b):
        return len(a) == len(b) and a == b

    def values_eq_approx(self, a, b):
        return len(a) == len(b) and all(float(x) == float(y) for (x, y) in zip(a, b))

    def c_support_code(self):
        return """
        #if PY_MAJOR_VERSION >= 3
            #ifndef PyInt_Check
                #define PyInt_Check PyLong_Check
            #endif
            #ifndef PyInt_AS_LONG
                #define PyInt_AS_LONG PyLong_AS_LONG
            #endif
        #endif
        """

    def c_declare(self, name, sub, check_input=True):
        return """
        %(ctype)s* %(name)s;
        """ % dict(ctype=self.ctype, name=name)

    def c_init(self, name, sub):
        return """
        %(name)s = NULL;
        """ % dict(name=name)

    def c_cleanup(self, name, sub):
        return """
        free(%(name)s);
        %(name)s = NULL;
        """ % dict(name=name)

    def c_extract(self, name, sub, check_input=True):
        return """
        ssize_t len;
        %(name)s = NULL;

        if (py_%(name)s == Py_None || !PyTuple_Check(py_%(name)s)) {
            PyErr_SetString(PyExc_TypeError, "CArrayFromTuple: expected tuple for extraction.");
            %(fail)s
        }
        len = PyTuple_GET_SIZE(py_%(name)s);
        %(name)s = (%(ctype)s*)malloc(len * sizeof(%(ctype)s));
        if (%(name)s == NULL) {
            PyErr_NoMemory();
            %(fail)s
        }
        for (ssize_t i = 0; i < len; ++i) {
            PyObject* element = PyTuple_GET_ITEM(py_%(name)s, i);
            if (PyInt_Check(element)) {
                %(name)s[i] = (%(ctype)s)PyInt_AS_LONG(element);
            } else if (PyFloat_Check(element)) {
                %(name)s[i] = (%(ctype)s)PyFloat_AS_DOUBLE(element);
            } else {
                PyErr_SetString(PyExc_TypeError, "CArrayFromTuple: expected integer or floating value to extract.");
            }
            Py_XDECREF(element);
            if (PyErr_Occurred()) {
                free(%(name)s);
                %(name)s = NULL;
                %(fail)s
            }
        }
        """ % dict(name=name, fail=sub['fail'], ctype=self.ctype)

    def c_code_cache_version(self):
        return (1, 1)


class CArrayType(Wrapper):
    """"
    Op parameter class to handle a Python vector of primitive types.
    You should better use this class instead of :class:`CArrayFromTuple`
    as it provides array length into C code by wrapping array data
    and array length into a Wrapper. Array C type is required when
    creating the type.

    .. code-block:: python

        carray_type = CArrayType('int')

        # In Python, it is easy to access length of an array.
        good_value = (1, 2, 3, 4, 5)
        len_good_value = len(good_value)

    In C code, a struct named CArrayType_<ctype> (here CArrayType_int) will be created
    with two field:

    - ``array``: C array with specified C type (``ctype* array;``).
    - ``length``: 64-bit signed integer which contains the number of elements into ``array`` (``npy_int64 length;``).

    .. code-block:: c

        CArrayType_int* good_value;
        /*
        good_value will be allocated and filled from its Python's `good_value` counterpart.
        ...
        */
        for (int i = 0; i < good_value->length; ++i) {
            printf("%d\n", good_value->array[i]);
        }

    """

    def __init__(self, ctype):
        self.carrayfromtuple = CArrayFromTuple(ctype)
        Wrapper.__init__(self, array=self.carrayfromtuple, length=theano.scalar.Scalar('int64'))

    def __repr__(self):
        return 'CArrayType<%s>' % self.carrayfromtuple.ctype

    def __eq__(self, other):
        return type(self) == type(other) and self.carrayfromtuple == other.carrayfromtuple

    def __hash__(self):
        return hash((type(self), self.carrayfromtuple))

    def generate_struct_name(self):
        return "CArrayType_" + self.carrayfromtuple.ctype.replace(' ', '_')

    def filter(self, data, strict=False, allow_downcast=None):
        return self.carrayfromtuple.filter(data, strict, allow_downcast)

    def values_eq(self, a, b):
        return self.carrayfromtuple.values_eq(a, b)

    def values_eq_approx(self, a, b):
        return self.carrayfromtuple.values_eq_approx(a, b)

    def c_code_cache_version(self):
        return (1, self.carrayfromtuple.c_code_cache_version(), Wrapper.c_code_cache_version(self))

    def c_support_code(self):
        return self.carrayfromtuple.c_support_code() + Wrapper.c_support_code(self)

    def c_extract(self, name, sub, check_input=True):
        return """
        %(name)s = new %(struct_name)s;
        %(name)s->extract_array(py_%(name)s);
        if (%(name)s->errorOccurred()) {
            %(fail)s
        }
        %(name)s->length = PyTuple_GET_SIZE(py_%(name)s);
        """ % dict(struct_name=self.name, name=name, fail=sub['fail'])
