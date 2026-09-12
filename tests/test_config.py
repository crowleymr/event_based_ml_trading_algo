from trading_pipeline.config import load_config


def test_locked_config():
    assert load_config("configs/poc.yaml")["horizon"] == 5
    names = open("configs/universe.txt").read().split()
    assert len(names) == len(set(names)) == 100
