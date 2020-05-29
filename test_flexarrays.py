import numpy as np
from flexarrays import *


def zero(*shape):
    return np.zeros(shape)

def one(*shape):
    return np.ones(shape)


def test_simple_vector():
    a = FlexArray(ndim=1)
    a._add_component('aa', zero(2))
    assert set(a) == {('aa',)}
    assert a.sizes == {'aa': 2}
    assert isinstance(a['aa'], np.ndarray)
    np.testing.assert_array_equal(a.realize[(R['aa'],)], zero(2))
    np.testing.assert_array_equal(a.realize[R['aa']], zero(2))
    np.testing.assert_array_equal(a.realize['aa'], zero(2))


def test_simple_matrix():
    a = FlexArray(ndim=2)
    a._add_component(('aa','aa'), zero(2,2))
    assert set(a) == {('aa','aa')}
    assert a.sizes == {'aa': 2}
    np.testing.assert_array_equal(a.realize[R['aa'], R['aa']], zero(2,2))
    np.testing.assert_array_equal(a.realize['aa', R['aa']], zero(2,2))
    np.testing.assert_array_equal(a.realize[R['aa'], 'aa'], zero(2,2))
    np.testing.assert_array_equal(a.realize['aa','aa'], zero(2,2))


def test_multi_vector():
    a = FlexArray(ndim=1)
    a._add_component('aa', zero(2))
    a._add_component('bb', one(3))
    assert set(a) == {('aa',), ('bb',)}
    assert a.sizes == {'aa': 2, 'bb': 3}
    np.testing.assert_array_equal(a.realize[R['aa','bb']], [*zero(2), *one(3)])
    np.testing.assert_array_equal(a.realize[R['bb','aa']], [*one(3), *zero(2)])
    np.testing.assert_array_equal(a.realize['aa'], zero(2))
    np.testing.assert_array_equal(a.realize['bb'], one(3))


def test_multi_matrix():
    a = FlexArray(ndim=2)
    a._add_component(('aa', 'aa'), zero(2,2))
    a._add_component(('aa', 'bb'), one(2,3))
    a._add_component(('bb', 'aa'), 2 * one(3,2))
    a._add_component(('bb', 'bb'), 3 * one(3,3))
    assert set(a) == {('aa','aa'), ('aa','bb'), ('bb','aa'), ('bb','bb')}
    assert a.sizes == {'aa': 2, 'bb': 3}
    np.testing.assert_array_equal(a.realize['aa','aa'], zero(2,2))
    np.testing.assert_array_equal(a.realize['aa','bb'], one(2,3))
    np.testing.assert_array_equal(a.realize['bb','aa'], 2 * one(3,2))
    np.testing.assert_array_equal(a.realize['bb','bb'], 3 * one(3,3))
    np.testing.assert_array_equal(a.realize[R['aa','bb'],'aa'], np.vstack([zero(2,2), 2*one(3,2)]))
    np.testing.assert_array_equal(a.realize['aa',R['aa','bb']], np.hstack([zero(2,2), one(2,3)]))
    np.testing.assert_array_equal(
        a.realize[R['aa','bb'],R['bb','aa']],
        np.block([[one(2,3), zero(2,2)], [3 * one(3,3), 2 * one(3,2)]])
    )


def test_slices():
    z = zero(1,1,1)
    a = FlexArray(ndim=3)
    a._add_component(('aa', 'zz', 'cc'), zero(1, 6, 3))
    a._add_component(('aa', 'bb', 'cc'), zero(1, 2, 3))
    a._add_component(('aa', 'dd', 'ee'), zero(1, 4, 5))
    assert a.sizes == {'aa': 1, 'bb': 2, 'cc': 3, 'dd': 4, 'ee': 5, 'zz': 6}

    b = a['aa', ...]
    assert b.ndim == 2
    assert set(b) == {('zz','cc'), ('bb','cc'), ('dd','ee')}
    assert b.sizes == {'bb': 2, 'cc': 3, 'dd': 4, 'ee': 5, 'zz': 6}

    b = a['aa']
    assert b.ndim == 2
    assert set(b) == {('zz','cc'), ('bb','cc'), ('dd','ee')}
    assert b.sizes == {'bb': 2, 'cc': 3, 'dd': 4, 'ee': 5, 'zz': 6}

    b = a['aa', 'zz', ...]
    assert b.ndim == 1
    assert set(b) == {('cc',)}
    assert b.sizes == {'cc': 3}


def test_numpy_contract():
    z = FlexArray(ndim=2)
    z['aa','aa'] = np.array([[1.0, 2.0], [3.0, 4.0]])
    z['aa','bb'] = np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
    z['bb','aa'] = np.array([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]])
    assert z.sizes == {'aa': 2, 'bb': 3}

    v = FlexArray(ndim=1)
    v['aa'] = np.array([17.0, 18.0])
    r = z.contract(v, axis=1)
    assert r.ndim == 1
    assert r.sizes == {'aa': 2, 'bb': 3}
    assert set(r) == {('aa',), ('bb',)}
    np.testing.assert_allclose(r['aa'], [53.0, 123.0])
    np.testing.assert_allclose(r['bb'], [403.0, 473.0, 543.0])

    v = FlexArray(ndim=2)
    v['aa','cc'] = np.array([[17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]])
    r = z.contract(v, axis=1)
    assert r.ndim == 2
    assert r.sizes == {'aa': 2, 'bb': 3, 'cc': 4}
    assert set(r) == {('aa','cc'), ('bb','cc')}
    np.testing.assert_allclose(r['aa','cc'], [[59.0, 62.0, 65.0, 68.0], [135.0, 142.0, 149.0, 156.0]])
    np.testing.assert_allclose(r['bb','cc'], [[439.0, 462.0, 485.0, 508.0],
                                              [515.0, 542.0, 569.0, 596.0],
                                              [591.0, 622.0, 653.0, 684.0]])
