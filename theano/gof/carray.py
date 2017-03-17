import re
import theano
from theano.gof import Type, Wrapper


class CArrayFromTuple(Type):
    """"
    Class for Op param to handle a Python vector of primitive types (integer, floating values or booleans).
    Booleans are converted to integers (False => 0, True -> 1).
    Vector should be represented as a tuple of values.

    In C, vectors are converted to a dynamically allocated C array.  C type of array elements is required to instanciate
    this type::

        carray_from_tuple = CArrayFromTuple('int')
        good_value = (1, 2, 3, 4, 5)

    In C code:

    .. code-block:: c

        int* good_value;
        /*
        good_value will be allocated and filled from its Python's `good_valus` counterpart.
        ...
        */
        int v = good_value[1]; // v == 2

    .. note::

        This class only create array in C, thus it does not provide array length into C code. You should better use
        :class:`CArrayType` which provide both array data and array length in C code.
    """

    def __init__(self, ctype):
        assert re.match('^[A-Za-z_][A-Za-z0-9_]*$', ctype), 'CArrayFromTuple: invalid C type name "%s".' % ctype
        self.ctype = ctype

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
        if (py_%(name)s == Py_None || !PyTuple_Check(py_%(name)s)) {
            PyErr_SetString(PyExc_TypeError, "CArrayFromTuple: expected tuple for extraction.");
            %(fail)s
        }
        ssize_t len = PyTuple_GET_SIZE(py_%(name)s);
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
        return (1,)


class CArrayType(Wrapper):
    """"
    Class for Op param handle a Python vector of primitive types.
    You should better use this class instead of :class:`CArrayFromTuple` because it provide array length into
    C code as it wraps array data and array length into a Wrapper.

    .. code-block:: python

        # Require C type.
        carray_type = CArrayType('int')

        # In Python, it is easy to access length of an array.
        good_value = (1, 2, 3, 4, 5)
        len_good_value = len(good_value)

    In C code, a struct named CArrayType_<ctype> (here CArrayType_int) will be created
    with two field:
    - ``array``, which points to the C array, with the right type.
    - ``length``, a 64-bit signed integer which contains the number of elements into ``array`` field.

    .. code-block:: c

        CArrayType_int* good_value;
        /*
        good_value will be allocated and filled from its Python's `good_valus` counterpart.
        ...
        */
        int len = good_value->length;
        for (int i = 0; i < len; ++i) {
            printf("%d\n", good_value->array[i]);
        }

    """

    def __init__(self, ctype):
        self.carrayfromtuple = CArrayFromTuple(ctype)
        Wrapper.__init__(self, array=self.carrayfromtuple, length=theano.scalar.Scalar('int64'))

    def __repr__(self):
        return 'CArrayType<%s>' % self.carrayfromtuple.ctype

    def __eq__(self, other):
        return type(self) == type(other) and self.carrayfromtuple == other.carraytype

    def __hash__(self):
        return hash((type(self), self.carrayfromtuple))

    def generate_struct_name(self):
        return "CArrayType_" + self.carrayfromtuple.ctype

    def filter(self, data, strict=False, allow_downcast=None):
        return self.carrayfromtuple.filter(data, strict, allow_downcast)

    def values_eq(self, a, b):
        return self.carrayfromtuple.values_eq(a, b)

    def values_eq_approx(self, a, b):
        return self.carrayfromtuple.values_eq_approx(a, b)

    def c_support_code(self):
        return self.carrayfromtuple.c_support_code() + Wrapper.c_support_code(self)

    def c_extract(self, name, sub, check_input=True):
        return """
        /* Seems c_init() is not called for a op param. So I call `new` here. */
        %(name)s = new %(struct_name)s;
        %(name)s->extract_array(py_%(name)s);
        if (PyErr_Occurred()) {
            %(fail)s
        }
        %(name)s->length = PyTuple_GET_SIZE(py_%(name)s);
        """ % dict(struct_name=self.name, name=name, fail=sub['fail'])
