from browser_env.processors import TextObervationProcessor


VIEWPORT = {"win_width": 100, "win_height": 100}


def test_element_fully_inside_viewport_ratio_is_one() -> None:
    ratio = TextObervationProcessor.get_element_in_viewport_ratio(
        elem_left_bound=10,
        elem_top_bound=10,
        width=20,
        height=30,
        config=VIEWPORT,
    )

    assert ratio == 1.0


def test_element_partially_inside_viewport_ratio_uses_element_area() -> None:
    ratio = TextObervationProcessor.get_element_in_viewport_ratio(
        elem_left_bound=90,
        elem_top_bound=80,
        width=20,
        height=40,
        config=VIEWPORT,
    )

    assert ratio == 0.25


def test_element_outside_viewport_ratio_is_zero() -> None:
    ratio = TextObervationProcessor.get_element_in_viewport_ratio(
        elem_left_bound=120,
        elem_top_bound=120,
        width=20,
        height=40,
        config=VIEWPORT,
    )

    assert ratio == 0.0


def test_zero_area_element_ratio_is_zero() -> None:
    ratio = TextObervationProcessor.get_element_in_viewport_ratio(
        elem_left_bound=10,
        elem_top_bound=10,
        width=0,
        height=40,
        config=VIEWPORT,
    )

    assert ratio == 0.0
