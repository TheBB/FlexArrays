from collections import namedtuple
from functools import partial, wraps, singledispatch
import inspect
from itertools import product
from multipledispatch import Dispatcher

import numpy as np
import scipy.sparse as sparselib


def is_compatible(index, slicer):
    for blockname, slc in zip(index, slicer):
        if isinstance(slc, str) and blockname != slc:
            return False
        if isinstance(slc, Range) and blockname not in slc:
            return False
    return True


def compatible_indexes(blocks, slicer):
    for index, value in blocks.items():
        if is_compatible(index, slicer):
            newindex = tuple(i for i, s in zip(index, slicer) if not isinstance(s, str))
            yield newindex, value


def expand_index(index, ndim):
    try:
        ellipsis_index = next(i for i, v in enumerate(index) if v == Ellipsis)
        assert ellipsis_index == len(index) - 1
        before, after = index[:ellipsis_index], index[ellipsis_index+1:]
    except StopIteration:
        before, after = index, ()
    nslices = ndim - len(before) - len(after)
    return (*before, *(slice(None) for _ in range(nslices)), *after)


def normalize_index(*argnames, expand=True):
    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            signature = inspect.signature(func)
            binding = signature.bind(*args, **kwargs)
            binding.apply_defaults()
            for argname in argnames:
                value = binding.arguments[argname]
                if isinstance(value, (str, Range)):
                    value = (value,)
                self = binding.arguments['self']
                if expand and hasattr(binding.arguments['self'], 'ndim') and self.ndim is not None:
                    value = expand_index(value, binding.arguments['self'].ndim)
                binding.arguments[argname] = value
            return func(*binding.args, **binding.kwargs)
        return inner
    return decorator


def normalize_multiindex(*argnames, expand=True):
    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            signature = inspect.signature(func)
            binding = signature.bind(*args, **kwargs)
            binding.apply_defaults()
            for argname in argnames:
                value = binding.arguments[argname]
                if isinstance(value, str):
                    value = (R[value],)
                elif isinstance(value, Range):
                    value = (value,)
                elif isinstance(value, tuple):
                    value = tuple(v if isinstance(v, Range) else R[v] for v in value)
                self = binding.arguments['self']
                if expand and hasattr(self, 'ndim') and self.ndim is not None:
                    value = expand_index(value, binding.arguments['self'].ndim)
                binding.arguments[argname] = value
            return func(*binding.args, **binding.kwargs)
        return inner
    return decorator


contract = Dispatcher('contract')

@contract.register(np.ndarray, np.ndarray, int)
def _(a, b, axis):
    if b.ndim == 1:
        return np.einsum(a, list(range(a.ndim)), b, [axis])
    out_order = list(range(a.ndim))
    out_order[axis] = a.ndim
    return np.einsum(a, list(range(a.ndim)), b, [axis, a.ndim], out_order)

@contract.register(sparselib.spmatrix, sparselib.spmatrix, int)
def _(a, b, axis):
    if axis == 1:
        return a.dot(b)
    return a.T.dot(b).T

@contract.register(sparselib.spmatrix, np.ndarray, int)
def _(a, b, axis):
    if axis == 1:
        return a.dot(b)
    return a.T.dot(b).T

@contract.register(np.ndarray, sparselib.spmatrix, int)
def _(a, b, axis):
    # Temporarily swap axes so that the target axis is in front
    a = np.swapaxes(a, 0, axis)
    retval = b.T.dot(a)
    retval = np.swapaxes(retval, 0, axis)
    return retval


@singledispatch
def copy(obj):
    raise NotImplementedError(type(obj))

@copy.register(np.ndarray)
def _(obj):
    return obj.copy()

@copy.register(sparselib.spmatrix)
def _(obj):
    return obj.copy()


@singledispatch
def transpose(obj, perm):
    raise NotImplementedError(type(obj))

@transpose.register(np.ndarray)
def _(obj, perm):
    return obj.transpose(perm)

@transpose.register(sparselib.spmatrix)
def _(obj, perm):
    if perm == (0, 1):
        return obj
    return obj.T


class Range:

    def __init__(self, blocknames):
        self.blocknames = blocknames

    def __iter__(self):
        yield from self.blocknames

    def __len__(self):
        return len(self.blocknames)

    def __contains__(self, blockname):
        return blockname in self.blocknames

class RangeBuilder:

    @normalize_index('blocknames')
    def __getitem__(self, blocknames):
        return Range(blocknames)

R = RangeBuilder()


class ZeroSentinel:

    @property
    def T(self):
        return self

    def __iadd__(self, other):
        return other

    def __add__(self, other):
        return other

    def __isub__(self, other):
        return -other

    def __sub__(self, other):
        return -other

    def __mul__(self, other):
        return self

    def __imul__(self, other):
        return self

    def __div__(self, other):
        return self

    def __idiv__(self, other):
        return self

zero_sentinel = ZeroSentinel()


class BlockDict(dict):

    @normalize_index('index')
    def __getitem__(self, index):
        return super().__getitem__(index)

    @normalize_index('index')
    def __setitem__(self, index, value):
        super().__setitem__(index, value)

    @normalize_index('index')
    def __delitem__(self, index):
        super().__delitem__(index)

    @normalize_index('index')
    def __contains__(self, index):
        return super().__contains__(index)

    def only(self):
        assert len(self) == 1
        return next(iter(self.items()))


class FlexArray(BlockDict):

    def __init__(self, ndim=None):
        super().__init__()
        self.sizes = dict()
        self.ndim = ndim

    @classmethod
    def vector(cls, index, value):
        """Simplified constructor for a single-block vector."""
        retval = cls(ndim=1)
        retval[(index,)] = value
        return retval

    @classmethod
    def single(cls, index, value):
        """Simplified constructor for a single-block array."""
        retval = cls(ndim=value.ndim)
        retval[index] = value
        return retval

    @property
    def realize(self):
        return Realizer(self)

    def ranges(self, blocks):
        previous, ranges = 0, []
        for block in blocks:
            ranges.append(np.arange(previous, previous + self.sizes[block]))
            previous += self.sizes[block]
        return ranges

    def keep(self, *blocks):
        retval = FlexArray(ndim=self.ndim)
        for index, value in self.items():
            if all(block in blocks for block in index):
                retval.add(index, value)
        return retval

    def discard(self, *blocks):
        retval = FlexArray(ndim=self.ndim)
        for index, value in self.items():
            if all(block not in blocks for block in index):
                retval.add(index, value)
        return retval

    @normalize_multiindex('indices', expand=False)
    def compatible(self, indices, array):
        ranges = [self.ranges(index) for index in indices]
        retval = FlexArray(ndim=array.ndim)
        retval.sizes.update(self.sizes)
        for index, blockranges in zip(product(*indices), product(*ranges)):
            retval[index] = array[np.ix_(*blockranges)]
        return retval

    @normalize_index('index')
    def add(self, index, value):
        self[index] = self.get(index, zero_sentinel) + value

    @normalize_index('index')
    def get(self, index, default=None, zero=None):
        if index in self:
            return super().__getitem__(index)
        if zero is not None:
            shape = tuple(self.sizes[blockname] for blockname in index)
            return zero(shape)
        return default

    @normalize_index('index')
    def __getitem__(self, index):
        if all(isinstance(blockname, str) for blockname in index):
            return super().__getitem__(index)
        new_ndim = sum(1 for blockname in index if not isinstance(blockname, str))
        retval = FlexArray(ndim=new_ndim)
        for newindex, value in compatible_indexes(self, index):
            retval.add(newindex, value)
        return retval

    @normalize_index('index')
    def __setitem__(self, index, value):
        if self.ndim is None:
            self.ndim = value.ndim
        for blockname, size in zip(index, value.shape[-self.ndim:]):
            assert size == self.sizes.setdefault(blockname, size)
        super().__setitem__(index, value)

    def copy(self, deep=True):
        retval = FlexArray(ndim=self.ndim)
        for index, value in self.items():
            if deep:
                value = copy(value)
            retval.add(index, value)
        return retval

    def transpose(self, perm):
        retval = FlexArray(ndim=self.ndim)
        for index, value in self.items():
            newindex = tuple(index[k] for k in perm)
            value = transpose(value, perm)
            retval.add(newindex, value)
        return retval

    @property
    def T(self):
        return self.transpose(tuple(reversed(range(self.ndim))))

    def contract(self, other, axis=None):
        # Normalize the axis index as a negative number, counting
        # backwards from the last axis.  This should correctly deal
        # with blocks which may have extra preceding dimensions.
        if axis is None:
            axis = self.ndim - 1
        posaxis = axis if axis >= 0 else axis + self.ndim
        negaxis = posaxis - self.ndim

        assert other.ndim in (1, 2)
        new_ndim = self.ndim + other.ndim - 2
        retval = FlexArray(ndim=new_ndim)

        for index, value in self.items():
            try:
                slc = other[index[axis]]
            except KeyError:
                continue
            contract_axis = value.ndim + negaxis
            if other.ndim == 1:
                newindex = (*index[:posaxis], *index[posaxis+1:])
                retval.add(newindex, contract(value, slc, contract_axis))
            elif other.ndim == 2:
                for newaxis, mx in slc.items():
                    newindex = (*index[:posaxis], *newaxis, *index[posaxis+1:])
                    retval.add(newindex, contract(value, mx, contract_axis))
        return retval

    def contract_many(self, contract):
        assert len(contract) == self.ndim
        retval = self.copy()
        for i, c in enumerate(reversed(contract)):
            if c is None:
                continue
            axis = self.ndim - i - 1
            retval = retval.contract(c, axis)
        return retval

    def __neg__(self):
        retval = FlexArray(ndim=self.ndim)
        for index, value in self.items():
            retval.add(index, -value)
        return retval

    def __iadd__(self, other):
        if np.isscalar(other):
            for index in self:
                self[index] += other
            return self
        if not isinstance(other, FlexArray):
            return NotImplemented
        if self.ndim is not None:
            assert self.ndim == other.ndim
        else:
            self.ndim = other.ndim
        for index, value in other.items():
            self.add(index, value)
        return self

    def __add__(self, other):
        if not np.isscalar(other) and not isinstance(other, FlexArray):
            return NotImplemented
        retval = self.copy()
        retval += other
        return retval

    def __radd__(self, other):
        return self.__add__(other)

    def __isub__(self, other):
        if np.isscalar(other):
            for index in self:
                self[index] -= other
            return self
        if not isinstance(other, FlexArray):
            return NotImplemented
        if self.ndim is not None:
            assert self.ndim == other.ndim
        else:
            self.ndim = other.ndim
        for index, value in other.items():
            self.add(index, -value)
        return self

    def __sub__(self, other):
        if not np.isscalar(other) and not isinstance(other, FlexArray):
            return NotImplemented
        retval = self.copy()
        retval -= other
        return retval

    def __rsub__(self, other):
        if not np.isscalar(other) and not isinstance(other, FlexArray):
            return NotImplemented
        return -self + other

    def __imul__(self, other):
        if np.isscalar(other):
            for index in self:
                self[index] *= other
            return self
        if not isinstance(other, FlexArray):
            return NotImplemented
        if self.ndim is not None:
            assert self.ndim == other.ndim
        else:
            self.ndim = other.ndim
        mine = set(self.keys())
        theirs = set(other.keys())
        for index in mine - theirs:
            del self[index]
        for index in mine & theirs:
            self[index] *= other[index]
        self.sizes = {**self.sizes, **other.sizes}
        return self

    def __mul__(self, other):
        if not np.isscalar(other) and not isinstance(other, FlexArray):
            return NotImplemented
        retval = self.copy()
        retval *= other
        return retval

    def __rmul__(self, other):
        return self * other


class Realizer:

    def __init__(self, array, sparse=None):
        self.array = array

        # If 'sparse' is not explicitly given, and we have a
        # two-dimensional block array with at least one sparse
        # component, return a sparse CSR matrix.  If 'sparse' is given
        # but does not specify the format, also use CSR.
        if sparse is None and array.ndim == 2:
            if any(isinstance(elt, sparselib.spmatrix) for elt in self.array.values()):
                self.sparse = 'csr'
            else:
                self.sparse = False
        elif array.ndim != 2:
            self.sparse = False
        elif sparse is True:
            self.sparse = 'csr'
        elif not sparse:
            self.sparse = False

    @property
    def ndim(self):
        return self.array.ndim

    def __call__(self, sparse=None):
        return Realizer(self.array, sparse)

    @normalize_multiindex('indices')
    def __getitem__(self, indices):
        array = self.array
        assert len(indices) == array.ndim

        blockshape = tuple(len(blocknames) for blocknames in indices)
        zerofunc = sparselib.coo_matrix if self.sparse else partial(np.full, fill_value=0.0)
        ordered_blocks = np.zeros(blockshape, dtype=object)
        for i, index in enumerate(product(*indices)):
            ordered_blocks.flat[i] = self.array.get(index, zero=zerofunc)

        if self.sparse:
            return sparselib.bmat(ordered_blocks, format=self.sparse)
        return np.block(ordered_blocks.tolist())
