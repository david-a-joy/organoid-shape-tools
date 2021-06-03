#!/usr/bin/env python3

# Standard lib
import unittest

# Our own imports
from organoid_shape_tools import utils

# Helpers


def init_one(x, y):
    cache = getattr(init_one, 'tricky_cache', [])
    cache.append((x, y))
    init_one.tricky_cache = cache


def add_one(x):
    return x + 1

# Tests


class TestHypermap(unittest.TestCase):

    def test_initializer_one_process(self):

        init_one.tricky_cache = []

        with utils.Hypermap(processes=1,
                            lazy=False,
                            initializer=init_one,
                            initargs=(1, 2)) as pool:
            res = pool.map(add_one, [1, 2, 3, 4, 5])

        exp = [2, 3, 4, 5, 6]

        self.assertEqual(res, exp)

        self.assertEqual(init_one.tricky_cache, [(1, 2)])

    def test_initializer_two_processes(self):

        # FIXME: This isn't a very good test, but it's unclear whether init can
        # be passed objects that survive in the parent context
        init_one.tricky_cache = []

        with utils.Hypermap(processes=2, lazy=False,
                            initializer=init_one,
                            initargs=(1, 2)) as pool:
            res = pool.map(add_one, [1, 2, 3, 4, 5])

        exp = [2, 3, 4, 5, 6]

        self.assertEqual(res, exp)

        self.assertEqual(init_one.tricky_cache, [])

    def test_map_one_process(self):

        with utils.Hypermap(processes=1, lazy=False) as pool:
            res = pool.map(add_one, [1, 2, 3, 4, 5])

        exp = [2, 3, 4, 5, 6]

        self.assertEqual(res, exp)

    def test_imap_one_process(self):

        with utils.Hypermap(processes=1, lazy=True) as pool:
            res = set(pool.map(add_one, [1, 2, 3, 4, 5]))

        exp = {2, 3, 4, 5, 6}

        self.assertEqual(res, exp)

    def test_map_two_processes(self):

        with utils.Hypermap(processes=2, lazy=False) as pool:
            res = pool.map(add_one, [1, 2, 3, 4, 5])

        exp = [2, 3, 4, 5, 6]

        self.assertEqual(res, exp)

    def test_imap_two_processes(self):

        with utils.Hypermap(processes=2, lazy=True) as pool:
            res = set(pool.map(add_one, [1, 2, 3, 4, 5]))

        exp = {2, 3, 4, 5, 6}

        self.assertEqual(res, exp)


if __name__ == '__main__':
    unittest.main()
