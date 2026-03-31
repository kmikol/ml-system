from shared.validation import validate_image

VALID = [[0.5] * 14 for _ in range(14)]  # 14x14 image, all values 0.5


def test_valid():
    assert validate_image(VALID) is None


def test_wrong_row_count():
    bad = [[0.5] * 14 for _ in range(10)]  # only 10 rows
    errors = validate_image(bad)
    assert errors is not None
    assert errors[0]["field"] == "image"


def test_wrong_col_count():
    bad = [[0.5] * 14 for _ in range(14)]
    bad[3] = [0.5] * 8  # row 3 has wrong width
    errors = validate_image(bad)
    assert errors is not None
    assert "image[3]" in errors[0]["field"]


def test_value_below_zero():
    bad = [row[:] for row in VALID]
    bad[0][0] = -0.1
    errors = validate_image(bad)
    assert errors is not None
    assert "image[0][0]" in errors[0]["field"]


def test_value_above_one():
    bad = [row[:] for row in VALID]
    bad[7][7] = 1.5
    errors = validate_image(bad)
    assert errors is not None
    assert "image[7][7]" in errors[0]["field"]


def test_not_a_list():
    errors = validate_image("not an image")
    assert errors is not None
    assert errors[0]["field"] == "image"


def test_boundary_values_valid():
    edge = [[0.0] * 14 for _ in range(14)]
    edge[0] = [1.0] * 14
    assert validate_image(edge) is None
