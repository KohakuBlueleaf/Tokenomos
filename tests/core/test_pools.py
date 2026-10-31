"""Slot and prefix entry sizing against a memory budget."""

from tokenomos.core import pools

MIB = 1 << 20


def test_explicit_counts_stay():
    assert pools.fit(104, 0, 400 * MIB, 0, MIB) == (104, 0)


def test_rings_bound_the_slots():
    slots, entries = pools.fit(None, None, 800 * MIB, 0, 6000 * MIB)
    assert slots == 7
    assert entries == 0


def test_a_cheap_model_reaches_the_ceiling():
    assert pools.fit(None, None, MIB, 8 * MIB, 20000 * MIB) == (256, 64)


def test_at_least_one_slot():
    assert pools.fit(None, None, 800 * MIB, 0, MIB)[0] == 1


def test_stateless_model():
    assert pools.fit(None, None, 0, 0, 0) == (256, 64)
