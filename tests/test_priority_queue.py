import random
import pytest

from neighbour_search import PriorityQueue


def test_ordered_ok():
    # setup
    pq = PriorityQueue()
    # stimuli
    pq.put(1, 1)
    pq.put(2, 2)
    pq.put(3, 3)
    # verify
    assert pq.pop() == 1
    assert pq.pop() == 2
    assert pq.pop() == 3
    # teardown
    pass


def test_reversed_ok():
    # setup
    pq = PriorityQueue()
    # stimuli
    pq.put(3, 3)
    pq.put(2, 2)
    pq.put(1, 1)
    # verify
    assert pq.pop() == 1
    assert pq.pop() == 2
    assert pq.pop() == 3
    # teardown
    pass


@pytest.fixture
def shuffled_priority_list():
    plist = [(i, i) for i in range(10)]
    random.shuffle(plist)

    return plist


def test_shuffled_ok(shuffled_priority_list):
    # setup
    pq = PriorityQueue()
    # stimuli
    for item, priority in shuffled_priority_list:
        pq.put(item, priority)
    # verify
    assert pq.pop() == 0
    # teardown
    pass


def test_empty_queue_ko():
    # setup
    pq = PriorityQueue()
    # stimuli
    pass
    # verify
    with pytest.raises(IndexError):
        pq.pop()
    # teardown
    pass
