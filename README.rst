==========
FlexArrays
==========

FlexArrays provides a library for arrays of delayed-realization block
type, that are symbolically indexed and internally unstructured.

This means:

- **Block type**: A FlexArray is a *block array*, similar in form and
  concept to
  `Numpy block arrays <https://numpy.org/doc/stable/reference/generated/numpy.block.html>`_ and
  and the `Scipy bmat function <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bmat.html>`_.

- **Delayed realization**: FlexArrays are not realized as true Numpy
  arrays or Scipy sparse matrices until specifically requested.

- **Symbolically indexed**: FlexArrays are indexed by *block*, and
  indices are strings rather than integers.  This means that you can
  use expressions like ``array['v','p']`` to refer to the
  velocity-pressure coupling block of a larger CFD system.

- **Internally unstructured**: FlexArray blocks are not ordered in any
  way, until realized as a Numpy array or a Scipy matrix.


How it works
------------

First, create a FlexArray.  The only information needed upon
construction is the number of dimensions.

.. code-block:: python

   from flexarrays import FlexArray

   array = FlexArray(ndim=2)


Then insert the blocks you need.

.. code-block:: python

   array['v','v'] = ...
   array['v','p'] = ...
   array['p','v'] = array['v','p'].T


The ``array`` object maintains an internal dictionary of all blocks,
and keeps track of the size of each named axis.  That means that all
*v*-dimensions and all *p*-dimensions must be the same size, no matter
where they appear.

In most situations, FlexArrays should behave as if any block not
explicitly stored is zero.  In this sense, FlexArrays are
block-sparse.  This applies to all arithmetic operations, but does not
apply to direct indexing.  Thus, ``array['p','p']`` will raise an
error rather than return something that can be interpreted as zero.
However, this means that an expression like ``array['p','p'] += ...``
will fail if that block is not yet stored.

The ``.add()`` method is a zero-tolerant construction method that
works around this.  In other words, the following are equivalent,
provided the ``('p','p')`` block does not yet exist:

.. code-block:: python

   array['p','p'] = a
   array['p','p'] += b

   array.add(('p','p'), a)
   array.add(('p','p'), b)


FlexArrays support the ordinary arithmetic operations involving
scalars and other FlexArrays, but because they are internally
unstructured, they cannot support such operations involving other
conventional array types, like Numpy arrays or Scipy sparse matrices.

To realize a conventional array type, use ``.realize[...]``.  For example

.. code-block:: python

   # Get the velocity-pressure block
   block = array.realize['v','p']


Of course, this could have been done with just ``array['v','p']``.
The ``.realize`` attribute is more useful when you want a range of
blocks.  To specify a range, it is recommended to use the ``R``
builder pattern:

.. code-block:: python

   from flexarrays import R

   # Get the vv and pv blocks
   block = array.realize[R['v','p'],'v']


This is sometimes necessary to avoid ambiguity.  An expression such as
``.realize['v','p']``, which is syntactically identical to
``.realize[('v','p')]`` is interpreted as a single block of a
two-dimensional array, rather than as two separate blocks of a
one-dimensional array.  While FlexArrays will conventionally interpret
tuples of strings as ranges in other contexts, such as
``.realize['v',('v','p')]``, we find it suitable to explicitly denote
ranges at all times using ``R``.

The ``.realize[...]`` operation will return a numpy array, unless at
least one sparse block is detected in the original data set.  This
holds *even if all the selected blocks are non-sparse*.

For explicit control, the ``.realize`` attribute is a callable object
that supports the ``sparse`` argument:

.. code-block:: python

   # Force sparseness
   array.realize(sparse=True)[...]

   # Force denseness
   array.realize(sparse=False)[...]

   # Force sparseness with a specific format
   array.realize(sparse='csr')[...]


Arithmetic and other array operations
-------------------------------------

FlexArrays support:

- addition, subtraction and multiplication involving FlexArrays and
  scalars, without broadcasting (number of dimensions must match)
- transposition using the ``.transpose()`` method and and ``.T``
  attribute
- rudimentary contraction, using ``array.contract(other, axis)`` to
  contract a given axis of ``array`` with the *first* axis of
  ``other``, which must be one- or two-dimensional

Other operations will be added when, and as required.
