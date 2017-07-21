# coding=utf-8
#
# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis-python
#
# Most of this work is copyright (C) 2013-2017 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others. See
# CONTRIBUTING.rst for a full list of people who may hold copyright, and
# consult the git log if you need to determine who owns an individual
# contribution.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.
#
# END HEADER

from __future__ import division, print_function, absolute_import

from hypothesis.internal.compat import hbytes, hrange


"""
This module implements a lexicographic minimizer for blocks of bytes.

That is, given a block of bytes of a given size, and a predicate that accepts
such blocks, it tries to find a lexicographically minimal block of that size
that satisfies the predicate, by repeatedly making local changes to that
starting point.

Assuming it is allowed to run to completion (which due to the way we use it it
actually often isn't) it makes the following guarantees, but it usually tries
to do better in practice:

1. The lexicographic predecessor (i.e. the largest block smaller than it) of
   the answer is not a solution.
2. No individual byte in the solution may be lowered while holding the others
   fixed.
"""


class Minimizer(object):

    def __init__(self, initial, condition, random):
        self.current = hbytes(initial)

        self.size = len(self.current)
        self.condition = condition
        self.random = random
        self.changes = 0
        self.seen = set()

    def incorporate(self, buffer):
        """Consider this buffer as a possible replacement for the current best
        buffer.

        Return True if it succeeds as such.

        """
        assert isinstance(buffer, hbytes)
        assert len(buffer) == self.size
        assert buffer <= self.current
        if buffer in self.seen:
            return False
        self.seen.add(buffer)
        if buffer != self.current and self.condition(buffer):
            self.current = buffer
            self.changes += 1
            return True
        return False

    def shift(self):
        """Attempt to shift individual byte values right as far as they can
        go."""
        prev = -1
        while prev != self.changes:
            prev = self.changes
            for i in hrange(self.size):
                block = bytearray(self.current)
                c = block[i]
                for k in hrange(c.bit_length(), 0, -1):
                    block[i] = c >> k
                    if self.incorporate(hbytes(block)):
                        break

    def replace(self, i, c):
        return self.current[:i] + hbytes([c]) + self.current[i + 1:]

    def rotate_suffixes(self):
        for significant, c in enumerate(self.current):  # pragma: no branch
            if c:
                break
        assert self.current[significant]

        prefix = hbytes(significant)

        for i in hrange(1, self.size - significant):
            left = self.current[significant:significant + i]
            right = self.current[significant + i:]
            rotated = prefix + right + left
            if rotated < self.current:
                self.incorporate(rotated)

    def shrink_indices(self):
        # We take a bet that there is some monotonic lower bound such that
        # whenever current >= lower_bound the result works.
        i = 0
        while i < self.size:
            if self.current[i] == 0 or self.incorporate(self.replace(i, 0)):
                i += 1
                continue

            prefix = self.current[:i]

            def suitable(c):
                """Does the lexicographically largest value starting with our
                prefix and having c at i satisfy the condition?"""
                if c == self.current[i]:
                    return True

                remainder = self.size - i - 1

                k = 0
                stopped = False
                while not stopped:
                    n = 2 ** k - 1
                    if n >= remainder:
                        stopped = True
                        n = remainder
                    k += 1
                    suffix = hbytes([255]) * n + self.current[i + n + 1:]
                    if self.incorporate(prefix + hbytes([c]) + suffix):
                        return True
                return False

            minimize_byte(self.current[i], suitable)
            i += 1

    def find_all_repeated_ngrams(self):
        index = {}
        for i, c in enumerate(self.current):
            index.setdefault(c, []).append(i)

        indices = [vs for vs in index.values() if len(vs) >= 2]
        length = 1
        results = set()
        seen_canon = set()
        while indices:
            new_indices = []
            for ix in indices:
                local_index = {}
                hit_limit = False
                for i in ix:
                    if i + length < self.size:
                        local_index.setdefault(
                            self.current[i + length], []).append(i)
                    else:
                        hit_limit = True
                for vs in local_index.values():
                    assert vs
                    if len(vs) < 2:
                        hit_limit = True
                    else:
                        new_indices.append(vs)
                if hit_limit and length > 1:
                    i = vs[0]

                    canonicalize = [i]
                    for j in ix:
                        if j >= canonicalize[-1] + length:
                            canonicalize.append(j)
                    if len(canonicalize) > 1:
                        token = self.current[i:i + length]
                        for i, c in enumerate(token):
                            if c:
                                offset = i
                                break
                        else:
                            continue
                        canonicalize = tuple(i + offset for i in canonicalize)
                        if canonicalize not in seen_canon:
                            results.add(token)
                            seen_canon.add(canonicalize)
            length += 1
            indices = new_indices
        return sorted(results, key=lambda s: (len(s), s), reverse=True)

    def minimize_repeated_tokens(self):
        i = 0
        local_changes = -1
        tokens = [None]

        while True:
            if self.changes != local_changes:
                tokens = self.find_all_repeated_ngrams()
                local_changes = self.changes
            if i >= len(tokens):
                break
            t = tokens[i]
            parts = self.current.split(t)
            assert len(parts) >= 2

            def token_condition(s):
                res = s.join(parts)
                if res == self.current:
                    return True
                else:
                    return self.incorporate(res)

            if len(t) == 2 or len(set(t)) > 1:
                minimize(
                    t,
                    token_condition, random=self.random,
                )
            i += 1

    def check_predecessor(self):
        predecessor = bytearray(self.current)
        for i in hrange(len(predecessor) - 1, -1, -1):
            if predecessor[i] == 0:
                predecessor[i] = 255
            else:
                predecessor[i] -= 1
                break
        else:
            assert False

        return self.incorporate(hbytes(predecessor))

    def alphabet_minimize(self):
        def cap(m):
            if m >= max(self.current):
                return True
            return self.incorporate(hbytes([min(m, c) for c in self.current]))

        minimize_byte(max(self.current), cap)

        for c in sorted(set(self.current), reverse=True):
            initial = self.current

            def replace(b):
                r = hbytes([b if d == c else d for d in initial])
                if r == self.current:
                    return True
                return self.incorporate(r)
            minimize_byte(c, replace)

    def sort_bytes(self):
        for i in hrange(self.size):
            j = i
            for j in hrange(i, 0, -1):
                if self.current[j - 1] <= self.current[j]:
                    break
                replacement = bytearray(self.current)
                replacement[j], replacement[j - 1] = replacement[j - 1], \
                    replacement[j]
                if not self.incorporate(hbytes(replacement)):
                    break

    def run(self):
        if not any(self.current):
            return

        # Initial checks as to whether the two smallest possible buffers of
        # this length can work. If so there's nothing to do here.
        if self.incorporate(hbytes(self.size)):
            return

        if self.incorporate(hbytes([0] * (self.size - 1) + [1])):
            return

        self.alphabet_minimize()

        if len(self.current) == 1:
            return

        # Perform a binary search to try to replace a long initial segment with
        # zero bytes.
        # Note that because this property isn't monotonic this will not always
        # find the longest subsequence we can replace with zero, only some
        # subsequence.

        # Replacing the first nonzero bytes with zero does *not* work
        nonzero = len(self.current)

        # Replacing the first canzero bytes with zero *does* work.
        canzero = 0
        while self.current[canzero] == 0:
            canzero += 1

        base = self.current

        @binsearch(canzero, nonzero)
        def zero_prefix(mid):
            return self.incorporate(
                hbytes(mid) +
                base[mid:]
            )

        base = self.current

        @binsearch(0, self.size)
        def shift_right(mid):
            if mid == 0:
                return True
            if mid == self.size:
                return False
            return self.incorporate(hbytes(mid) + base[:-mid])

        self.minimize_repeated_tokens()

        self.sort_bytes()

        if self.check_predecessor():
            self.shrink_indices()


def minimize(initial, condition, random):
    """Perform a lexicographical minimization of the byte string 'initial' such
    that the predicate 'condition' returns True, and return the minimized
    string."""
    m = Minimizer(initial, condition, random)
    m.run()
    return m.current


def binsearch(_lo, _hi):
    """Run a binary search to find the point at which a function changes value
    between two bounds.

    This function is used purely for its side effects and returns
    nothing.

    """
    def accept(f):
        lo = _lo
        hi = _hi

        loval = f(lo)
        hival = f(hi)

        if loval == hival:
            return

        while lo + 1 < hi:
            mid = (lo + hi) // 2
            midval = f(mid)
            if midval == loval:
                lo = mid
            else:
                assert hival == midval
                hi = mid
    return accept


def minimize_byte(c, f):
    if c == 0 or f(0):
        return 0
    elif c == 1 or f(1):
        return 1
    elif c == 2:
        return 2
    if not f(c - 1):
        return c

    lo = 1
    hi = c - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi
