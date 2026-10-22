from split_cloth.sampling import page_groups, sample_indices


def test_timing_and_tail():
    assert sample_indices(1, 20) == [0]
    assert sample_indices(21, 20) == [0, 20]
    assert sample_indices(26, 20) == [0, 20, 25]
    assert sample_indices(31, 10) == [0, 10, 20, 30]


def test_page_overlap_without_redundant_tail():
    assert page_groups(12) == [list(range(12))]
    assert page_groups(13) == [list(range(12)), list(range(10, 13))]
    assert page_groups(22) == [list(range(12)), list(range(10, 22))]
    assert page_groups(23)[-1] == [20, 21, 22]
