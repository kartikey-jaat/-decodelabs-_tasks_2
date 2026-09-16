from src.data_loader import load_data


def test_load_data():
    X, y, feature_names, target_names = load_data()

    assert X.shape == (150, 4)
    assert len(y) == 150
    assert len(feature_names) == 4
    assert len(target_names) == 3