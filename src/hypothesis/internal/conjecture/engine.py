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

import time
from enum import Enum
from random import Random, getrandbits
from weakref import WeakKeyDictionary

from hypothesis import settings as Settings
from hypothesis import Phase
from hypothesis.reporting import debug_report
from hypothesis.internal.compat import EMPTY_BYTES, Counter, hbytes, \
    hrange, text_type, bytes_from_list, to_bytes_sequence, \
    unicode_safe_repr
from hypothesis.internal.conjecture.data import MAX_DEPTH, Status, \
    StopTest, ConjectureData
from hypothesis.internal.conjecture.minimizer import minimize


class ExitReason(Enum):
    max_examples = 0
    max_iterations = 1
    timeout = 2
    max_shrinks = 3
    finished = 4
    flaky = 5


class RunIsComplete(Exception):
    pass


class ConjectureRunner(object):

    def __init__(
        self, test_function, settings=None, random=None,
        database_key=None,
    ):
        self._test_function = test_function
        self.settings = settings or Settings()
        self.last_data = None
        self.changed = 0
        self.shrinks = 0
        self.call_count = 0
        self.event_call_counts = Counter()
        self.valid_examples = 0
        self.start_time = time.time()
        self.random = random or Random(getrandbits(128))
        self.database_key = database_key
        self.status_runtimes = {}
        self.events_to_strings = WeakKeyDictionary()

        # Tree nodes are stored in an array to prevent heavy nesting of data
        # structures. Branches are dicts mapping bytes to child nodes (which
        # will in general only be partially populated). Leaves are
        # ConjectureData objects that have been previously seen as the result
        # of following that path.
        self.tree = [{}, None]

        # A node is dead if there is nothing left to explore past that point.
        # Recursively, a node is dead if either it is a leaf or every byte
        # leads to a dead node when starting from here.
        self.dead = set()
        self.forced = {}

    def __tree_is_exhausted(self):
        return 0 in self.dead

    def new_buffer(self):
        assert not self.__tree_is_exhausted()

        def draw_bytes(data, n, distribution):
            return self.__rewrite_for_novelty(
                data, self.__zero_bound(data, distribution(self.random, n))
            )

        self.last_data = ConjectureData(
            max_length=self.settings.buffer_size,
            draw_bytes=draw_bytes
        )
        self.test_function(self.last_data)
        self.last_data.freeze()

    def test_function(self, data):
        self.call_count += 1
        try:
            self._test_function(data)
            data.freeze()
        except StopTest as e:
            if e.testcounter != data.testcounter:
                self.save_buffer(data.buffer)
                raise e
        except:
            self.save_buffer(data.buffer)
            raise
        finally:
            data.freeze()
            self.note_details(data)

        self.debug_data(data)
        if data.status >= Status.VALID:
            self.valid_examples += 1

        tree_node = self.tree[0]
        indices = []
        node_index = 0
        for i, b in enumerate(data.buffer):
            indices.append(node_index)
            if i in data.forced_indices:
                self.forced[node_index] = b
            try:
                node_index = tree_node[b]
            except KeyError:
                node_index = len(self.tree)
                self.tree.append({})
                tree_node[b] = node_index
            tree_node = self.tree[node_index]
            if node_index in self.dead:
                break

        if data.status != Status.OVERRUN and node_index not in self.dead:
            self.dead.add(node_index)
            self.tree[node_index] = data

            for j in reversed(indices):
                if len(self.tree[j]) < 256 and j not in self.forced:
                    break
                if set(self.tree[j].values()).issubset(self.dead):
                    self.dead.add(j)
                else:
                    break

    def consider_new_test_data(self, data):
        # Transition rules:
        #   1. Transition cannot decrease the status
        #   2. Any transition which increases the status is valid
        #   3. If the previous status was interesting, only shrinking
        #      transitions are allowed.
        if data.buffer == self.last_data.buffer:
            return False
        if self.last_data.status < data.status:
            return True
        if self.last_data.status > data.status:
            return False
        if data.status == Status.INVALID:
            return data.index >= self.last_data.index
        if data.status == Status.OVERRUN:
            return data.overdraw <= self.last_data.overdraw
        if data.status == Status.INTERESTING:
            assert len(data.buffer) <= len(self.last_data.buffer)
            if len(data.buffer) == len(self.last_data.buffer):
                return data.buffer < self.last_data.buffer
            return True
        return True

    def save_buffer(self, buffer):
        if (
            self.settings.database is not None and
            self.database_key is not None and
            Phase.reuse in self.settings.phases
        ):
            self.settings.database.save(
                self.database_key, hbytes(buffer)
            )

    def note_details(self, data):
        if data.status == Status.INTERESTING:
            self.save_buffer(data.buffer)
        runtime = max(data.finish_time - data.start_time, 0.0)
        self.status_runtimes.setdefault(data.status, []).append(runtime)
        for event in set(map(self.event_to_string, data.events)):
            self.event_call_counts[event] += 1

    def debug(self, message):
        with self.settings:
            debug_report(message)

    def debug_data(self, data):
        self.debug(u'%d bytes %s -> %s, %s' % (
            data.index,
            unicode_safe_repr(list(data.buffer[:data.index])),
            unicode_safe_repr(data.status),
            data.output,
        ))

    def prescreen_buffer(self, buffer):
        i = 0
        for b in buffer:
            if i in self.dead:
                return False
            try:
                b = self.forced[i]
            except KeyError:
                pass
            try:
                i = self.tree[i][b]
            except KeyError:
                return True
        else:
            return False

    def incorporate_new_buffer(self, buffer):
        assert self.last_data.status == Status.INTERESTING
        if (
            self.settings.timeout > 0 and
            time.time() >= self.start_time + self.settings.timeout
        ):
            self.exit_reason = ExitReason.timeout
            raise RunIsComplete()

        buffer = hbytes(buffer[:self.last_data.index])
        if sort_key(buffer) >= sort_key(self.last_data.buffer):
            return False

        if not self.prescreen_buffer(buffer):
            return False

        assert sort_key(buffer) <= sort_key(self.last_data.buffer)
        data = ConjectureData.for_buffer(buffer)
        self.test_function(data)
        if self.consider_new_test_data(data):
            self.shrinks += 1
            self.last_data = data
            if self.shrinks >= self.settings.max_shrinks:
                self.exit_reason = ExitReason.max_shrinks
                raise RunIsComplete()
            self.last_data = data
            self.changed += 1
            return True
        return False

    def run(self):
        with self.settings:
            try:
                self._run()
            except RunIsComplete:
                pass
            self.debug(
                u'Run complete after %d examples (%d valid) and %d shrinks' % (
                    self.call_count, self.valid_examples, self.shrinks,
                ))

    def _new_mutator(self):
        def draw_new(data, n, distribution):
            return distribution(self.random, n)

        def draw_existing(data, n, distribution):
            return self.last_data.buffer[data.index:data.index + n]

        def draw_smaller(data, n, distribution):
            existing = self.last_data.buffer[data.index:data.index + n]
            r = distribution(self.random, n)
            if r <= existing:
                return r
            return _draw_predecessor(self.random, existing)

        def draw_larger(data, n, distribution):
            existing = self.last_data.buffer[data.index:data.index + n]
            r = distribution(self.random, n)
            if r >= existing:
                return r
            return _draw_successor(self.random, existing)

        def reuse_existing(data, n, distribution):
            choices = data.block_starts.get(n, []) or \
                self.last_data.block_starts.get(n, [])
            if choices:
                i = self.random.choice(choices)
                return self.last_data.buffer[i:i + n]
            else:
                result = distribution(self.random, n)
                assert isinstance(result, hbytes)
                return result

        def flip_bit(data, n, distribution):
            buf = bytearray(
                self.last_data.buffer[data.index:data.index + n])
            i = self.random.randint(0, n - 1)
            k = self.random.randint(0, 7)
            buf[i] ^= (1 << k)
            return hbytes(buf)

        def draw_zero(data, n, distribution):
            return hbytes(b'\0' * n)

        def draw_max(data, n, distribution):
            return hbytes([255]) * n

        def draw_constant(data, n, distribution):
            return bytes_from_list([
                self.random.randint(0, 255)
            ] * n)

        options = [
            draw_new,
            reuse_existing, reuse_existing,
            draw_existing, draw_smaller, draw_larger,
            flip_bit,
            draw_zero, draw_max, draw_zero, draw_max,
            draw_constant,
        ]

        bits = [
            self.random.choice(options) for _ in hrange(3)
        ]

        def draw_mutated(data, n, distribution):
            if (
                data.index + n > len(self.last_data.buffer)
            ):
                result = distribution(self.random, n)
            else:
                result = self.random.choice(bits)(data, n, distribution)

            return self.__rewrite_for_novelty(
                data, self.__zero_bound(data, result))

        return draw_mutated

    def __rewrite(self, data, result):
        return self.__rewrite_for_novelty(
            data, self.__zero_bound(data, result)
        )

    def __zero_bound(self, data, result):
        """This tries to get the size of the generated data under control by
        replacing the result with zero if we are too deep or have already
        generated too much data.

        This causes us to enter "shrinking mode" there and thus reduce
        the size of the generated data.

        """
        if (
            data.depth * 2 >= MAX_DEPTH or
            (data.index + len(result)) * 2 >= self.settings.buffer_size
        ):
            if any(result):
                data.hit_zero_bound = True
            return hbytes(len(result))
        else:
            return result

    def __rewrite_for_novelty(self, data, result):
        """Take a block that is about to be added to data as the result of a
        draw_bytes call and rewrite it a small amount to ensure that the result
        will be novel: that is, not hit a part of the tree that we have fully
        explored.

        This is mostly useful for test functions which draw a small
        number of blocks.

        """
        assert isinstance(result, hbytes)
        try:
            node_index = data.__current_node_index
        except AttributeError:
            node_index = 0
            data.__current_node_index = node_index
            data.__hit_novelty = False
            data.__evaluated_to = 0

        if data.__hit_novelty:
            return result

        node = self.tree[node_index]

        for i in hrange(data.__evaluated_to, len(data.buffer)):
            node = self.tree[node_index]
            try:
                node_index = node[data.buffer[i]]
                assert node_index not in self.dead
                node = self.tree[node_index]
            except KeyError:
                data.__hit_novelty = True
                return result

        for i, b in enumerate(result):
            assert isinstance(b, int)
            try:
                new_node_index = node[b]
            except KeyError:
                data.__hit_novelty = True
                return result

            new_node = self.tree[new_node_index]

            if new_node_index in self.dead:
                if isinstance(result, hbytes):
                    result = bytearray(result)
                for c in range(256):
                    if c not in node:
                        result[i] = c
                        data.__hit_novelty = True
                        return hbytes(result)
                    else:
                        new_node_index = node[c]
                        new_node = self.tree[new_node_index]
                        if new_node_index not in self.dead:
                            result[i] = c
                            break
                else:  # pragma: no cover
                    assert False, (
                        'Found a tree node which is live despite all its '
                        'children being dead.')
            node_index = new_node_index
            node = new_node
        assert node_index not in self.dead
        data.__current_node_index = node_index
        data.__evaluated_to = data.index + len(result)
        return hbytes(result)

    def _run(self):
        self.last_data = None
        mutations = 0
        start_time = time.time()

        if (
            self.settings.database is not None and
            self.database_key is not None
        ):
            corpus = sorted(
                self.settings.database.fetch(self.database_key),
                key=lambda d: (len(d), d)
            )
            for existing in corpus:
                if self.valid_examples >= self.settings.max_examples:
                    self.exit_reason = ExitReason.max_examples
                    return
                if self.call_count >= max(
                    self.settings.max_iterations, self.settings.max_examples
                ):
                    self.exit_reason = ExitReason.max_iterations
                    return
                data = ConjectureData.for_buffer(existing)
                self.test_function(data)
                data.freeze()
                self.last_data = data
                self.consider_new_test_data(data)
                if data.status < Status.VALID:
                    self.settings.database.delete(
                        self.database_key, existing)
                elif data.status == Status.VALID:
                    # Incremental garbage collection! we store a lot of
                    # examples in the DB as we shrink: Those that stay
                    # interesting get kept, those that become invalid get
                    # dropped, but those that are merely valid gradually go
                    # away over time.
                    if self.random.randint(0, 2) == 0:
                        self.settings.database.delete(
                            self.database_key, existing)
                else:
                    assert data.status == Status.INTERESTING
                    self.last_data = data
                    break

        if (
            Phase.generate in self.settings.phases and not
            self.__tree_is_exhausted()
        ):
            if (
                self.last_data is None or
                self.last_data.status < Status.INTERESTING
            ):
                self.new_buffer()

            mutator = self._new_mutator()

            zero_bound_queue = []

            while (
                self.last_data.status != Status.INTERESTING and
                not self.__tree_is_exhausted()
            ):
                if self.valid_examples >= self.settings.max_examples:
                    self.exit_reason = ExitReason.max_examples
                    return
                if self.call_count >= max(
                    self.settings.max_iterations, self.settings.max_examples
                ):
                    self.exit_reason = ExitReason.max_iterations
                    return
                if (
                    self.settings.timeout > 0 and
                    time.time() >= start_time + self.settings.timeout
                ):
                    self.exit_reason = ExitReason.timeout
                    return
                if zero_bound_queue:
                    # Whenever we generated an example and it hits a bound
                    # which forces zero blocks into it, this creates a weird
                    # distortion effect by making certain parts of the data
                    # stream (especially ones to the right) much more likely
                    # to be zero. We fix this by redistributing the generated
                    # data by shuffling it randomly. This results in the
                    # zero data being spread evenly throughout the buffer.
                    # Hopefully the shrinking this causes will cause us to
                    # naturally fail to hit the bound.
                    # If it doesn't then we will queue the new version up again
                    # (now with more zeros) and try again.
                    overdrawn = zero_bound_queue.pop()
                    buffer = bytearray(overdrawn.buffer)
                    self.random.shuffle(buffer)
                    buffer = hbytes(buffer)

                    if buffer == overdrawn.buffer:
                        continue

                    def draw_bytes(data, n, distribution):
                        result = buffer[data.index:data.index + n]
                        if len(result) < n:
                            result += hbytes(n - len(result))
                        return self.__rewrite(data, result)

                    data = ConjectureData(
                        draw_bytes=draw_bytes,
                        max_length=self.settings.buffer_size,
                    )
                    self.test_function(data)
                    data.freeze()
                elif mutations >= self.settings.max_mutations:
                    mutations = 0
                    data = self.new_buffer()
                    mutator = self._new_mutator()
                else:
                    data = ConjectureData(
                        draw_bytes=mutator,
                        max_length=self.settings.buffer_size
                    )
                    self.test_function(data)
                    data.freeze()
                    prev_data = self.last_data
                    if self.consider_new_test_data(data):
                        self.last_data = data
                        if data.status > prev_data.status:
                            mutations = 0
                    else:
                        mutator = self._new_mutator()
                if getattr(data, 'hit_zero_bound', False):
                    zero_bound_queue.append(data)
                mutations += 1

        if self.__tree_is_exhausted():
            self.exit_reason = ExitReason.finished
            return

        data = self.last_data
        if data is None:
            self.exit_reason = ExitReason.finished
            return
        assert isinstance(data.output, text_type)

        if self.settings.max_shrinks <= 0:
            self.exit_reason = ExitReason.max_shrinks
            return

        if Phase.shrink not in self.settings.phases:
            self.exit_reason = ExitReason.finished
            return

        data = ConjectureData.for_buffer(self.last_data.buffer)
        self.test_function(data)
        if data.status != Status.INTERESTING:
            self.exit_reason = ExitReason.flaky
            return

        self.shrink()

    def is_shrink_point(self, i):
        """A shrink point is a block that when lexicographically minimized will
        cause the size of the generated example to shrink (while remaining
        valid)."""

        if i >= len(self.last_data.blocks):
            return False
        u, v = self.last_data.blocks[i]
        block = self.last_data.buffer[u:v]
        buf = self.last_data.buffer
        i += 1
        if not any(block):
            return False
        zeroed = buf[:u] + hbytes(v - u) + buf[v:]

        def draw_bytes(data, n, distribution):
            result = zeroed[data.index:data.index + n]
            if len(result) < n:
                result += hbytes(n - len(result))
            return self.__rewrite(data, result)

        zeroed_data = ConjectureData(
            draw_bytes=draw_bytes, max_length=len(zeroed)
        )
        self.test_function(zeroed_data)

        # We can just zero the whole block. Nothing left to do here.
        if zeroed_data.status == Status.INTERESTING:
            if (
                sort_key(zeroed_data.buffer) <
                sort_key(self.last_data.buffer)
            ):
                self.last_data = zeroed_data
            return False

        # If zeroing it produced an invalid example then this is probably
        # not a good shrink point. We might lose some useful shrinks, but
        # it's nothing to be too worried about if we do and it doesn't
        # generally happen in the main case we're interested in.
        if zeroed_data.status < Status.VALID:
            return False

        # It didn't reduce the size when we zeroed it, so it's not a
        # shrink point.
        return zeroed_data.index < len(buf)

    def minimize_at_shrink_points(self):
        """This method handles the case where shrinking a block can cause the
        amount of data that is drawn to decrease. It's primarily concerned with
        the case where someone has drawn an integer and then drawn that many
        elements, but it may be useful in other cases too.

        The big problem with this scenario is that we can end up with the
        interesting elements at the end of the list, which can prevent us from
        shrinking purely because we can only shrink by reducing the length
        parameter, which amounts to taking prefixes.

        This attempts to fix that problem by trying a number of rearrangements
        of the suffix while shrinking when it occurs, that can allow us to
        reduce the parameter by moving the intersting elements earlier.

        """

        self.debug('Identifying and minimizing shrink points')

        i = 0
        while i < len(self.last_data.blocks):
            # Although shrink points are blocks, it may be that we have
            # multiple shrink points next to eachother. When this happens we
            # want to merge them into a single interval to shrink all at once
            # as otherwise this step can get very expensive.

            if not self.is_shrink_point(i):
                i += 1
                continue
            j = i + 1
            while self.is_shrink_point(j):
                j += 1

            u = self.last_data.blocks[i][0]
            v = self.last_data.blocks[j][0]

            buf = self.last_data.buffer

            i = j

            self.debug('Identified shrink point %r at %i' % (
                buf[u:v], i
            ))

            prefix = buf[:u]

            # It is now reasonably minimized, but is blocked from going further
            # by whatever is in the rest of the data. We now try a more
            # expensive predicate to minimize by.

            def check(value):
                """Tried value as a replacement for block, rearranging the
                suffix a number of times to attempt to make it work if needs
                be."""

                initial_attempt = prefix + value + \
                    self.last_data.buffer[v:]

                initial_data = None
                node_index = 0
                for c in initial_attempt:
                    try:
                        node_index = self.tree[node_index][c]
                    except KeyError:
                        break
                    node = self.tree[node_index]
                    if isinstance(node, ConjectureData):
                        initial_data = node
                        break

                if initial_data is None:
                    initial_data = ConjectureData.for_buffer(initial_attempt)
                    self.test_function(initial_data)

                if initial_data.status == Status.INTERESTING:
                    self.last_data = initial_data
                    return True

                # If this produced something completely invalid we ditch it
                # here rather than trying to persevere.
                if initial_data.status < Status.VALID:
                    return False

                lost_data = len(self.last_data.buffer) - \
                    len(initial_data.buffer)

                # If this did not in fact cause the data size to shrink we
                # bail here because it's not worth trying to delete stuff from
                # the remainder.
                if not lost_data:
                    return False

                try_with_deleted = bytearray(self.last_data.buffer)
                try_with_deleted[u:v] = value
                del try_with_deleted[v:v + lost_data]
                if self.incorporate_new_buffer(try_with_deleted):
                    return True
                seen_size = set()
                for r, s in self.last_data.blocks[i:]:
                    buf = self.last_data.buffer
                    size = s - r
                    if (
                        not any(buf[r:s]) and size <= lost_data
                        and size not in seen_size
                    ):
                        seen_size.add(size)
                        trial = bytearray(buf)
                        trial[u:v] = value
                        del trial[r:s]
                        if self.incorporate_new_buffer(trial):
                            return True
                return False
            minimize(self.last_data.buffer[u:v], check, random=self.random)

    def greedy_interval_deletion(self):
        """Attempt to delete every interval in the example."""

        self.debug('Structured interval deletes')

        # We do a delta-debugging style thing here where we initially try to
        # delete many intervals at once and prune it down exponentially to
        # eventually only trying to delete one interval at a time.

        # I'm a little skeptical that this is helpful in general, but we've
        # got at least one benchmark where it does help.
        k = len(self.last_data.intervals) // 2
        while k > 0:
            i = 0
            while i + k <= len(self.last_data.intervals):
                bitmask = [True] * len(self.last_data.buffer)

                for u, v in self.last_data.intervals[i:i + k]:
                    for t in range(u, v):
                        bitmask[t] = False

                u, v = self.last_data.intervals[i]
                if not self.incorporate_new_buffer(hbytes(
                    b for b, v in zip(self.last_data.buffer, bitmask)
                    if v
                )):
                    i += k
            k //= 2

    def minimize_duplicated_blocks(self):
        """Find blocks that have been duplicated in multiple places and attempt
        to minimize all of the duplicates simultaneously."""

        self.debug('Simultaneous shrinking of duplicated blocks')
        counts = Counter(
            self.last_data.buffer[u:v] for u, v in self.last_data.blocks
        )
        blocks = [
            k for k, count in
            counts.items()
            if count > 1
        ]
        blocks.sort(reverse=True)
        blocks.sort(key=lambda b: counts[b] * len(b), reverse=True)
        for block in blocks:
            parts = [
                self.last_data.buffer[r:s]
                for r, s in self.last_data.blocks
            ]

            def replace(b):
                return hbytes(EMPTY_BYTES.join(
                    hbytes(b if c == block else c) for c in parts
                ))
            minimize(
                block,
                lambda b: self.incorporate_new_buffer(replace(b)),
                random=self.random,
            )

    def minimize_individual_blocks(self):
        self.debug('Shrinking of individual blocks')
        i = 0
        while i < len(self.last_data.blocks):
            u, v = self.last_data.blocks[i]
            minimize(
                self.last_data.buffer[u:v],
                lambda b: self.incorporate_new_buffer(
                    self.last_data.buffer[:u] + b +
                    self.last_data.buffer[v:],
                ),
                random=self.random,
            )
            i += 1

    def reorder_blocks(self):
        self.debug('Reordering blocks')
        block_lengths = sorted(self.last_data.block_starts, reverse=True)
        for n in block_lengths:
            i = 1
            while i < len(self.last_data.block_starts.get(n, ())):
                j = i
                while j > 0:
                    buf = self.last_data.buffer
                    blocks = self.last_data.block_starts[n]
                    a_start = blocks[j - 1]
                    b_start = blocks[j]
                    a = buf[a_start:a_start + n]
                    b = buf[b_start:b_start + n]
                    if a <= b:
                        break
                    swapped = (
                        buf[:a_start] + b + buf[a_start + n:b_start] +
                        a + buf[b_start + n:])
                    assert len(swapped) == len(buf)
                    assert swapped < buf
                    if self.incorporate_new_buffer(swapped):
                        j -= 1
                    else:
                        break
                i += 1

    def shrink(self):
        # We assume that if an all-zero block of bytes is an interesting
        # example then we're not going to do better than that.
        # This might not technically be true: e.g. for integers() | booleans()
        # the simplest example is actually [1, 0]. Missing this case is fairly
        # harmless and this allows us to make various simplifying assumptions
        # about the structure of the data (principally that we're never
        # operating on a block of all zero bytes so can use non-zeroness as a
        # signpost of complexity).
        if (
            not any(self.last_data.buffer) or
            self.incorporate_new_buffer(hbytes(len(self.last_data.buffer)))
        ):
            self.exit_reason = ExitReason.finished
            return

        change_counter = -1

        while self.changed > change_counter:
            change_counter = self.changed

            self.minimize_at_shrink_points()
            self.greedy_interval_deletion()

            self.minimize_duplicated_blocks()
            self.minimize_individual_blocks()

            self.reorder_blocks()

        self.exit_reason = ExitReason.finished

    def event_to_string(self, event):
        if isinstance(event, str):
            return event
        try:
            return self.events_to_strings[event]
        except KeyError:
            pass
        result = str(event)
        self.events_to_strings[event] = result
        return result


def _draw_predecessor(rnd, xs):
    r = bytearray()
    any_strict = False
    for x in to_bytes_sequence(xs):
        if not any_strict:
            c = rnd.randint(0, x)
            if c < x:
                any_strict = True
        else:
            c = rnd.randint(0, 255)
        r.append(c)
    return hbytes(r)


def _draw_successor(rnd, xs):
    r = bytearray()
    any_strict = False
    for x in to_bytes_sequence(xs):
        if not any_strict:
            c = rnd.randint(x, 255)
            if c > x:
                any_strict = True
        else:
            c = rnd.randint(0, 255)
        r.append(c)
    return hbytes(r)


def sort_key(buffer):
    return (len(buffer), buffer)
