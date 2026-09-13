import numpy as np
from trading_pipeline.modelling.training import estimator


def test_preprocessing_training_only():
    x = np.array([[1., 3.], [2., np.nan], [3., 5.], [4., 6.]])
    model = estimator("elastic_net", {"alpha": .001, "l1_ratio": .5}, 42).fit(x, np.arange(4.))
    impute = model["imputer"].statistics_.copy()
    means = model["scaler"].mean_.copy()
    model.predict(np.array([[1e12, np.nan]]))
    np.testing.assert_array_equal(impute, model["imputer"].statistics_)
    np.testing.assert_array_equal(means, model["scaler"].mean_)
    assert means[0] == 2.5
