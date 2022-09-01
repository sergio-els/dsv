from sklearn.datasets import make_classification
from sklearn.svm import SVC


def test_sv_then_weights():
    for i in range(0, 100):
        test_sv_then_weights_try()


def test_sv_then_weights_try():
    # create a sample set which is linearly separable
    sample = make_classification(n_samples=1000, class_sep=1, n_features=4, n_redundant=2, n_informative=2,
                                 n_clusters_per_class=1, flip_y=0)
    xx = sample[0]
    y = sample[1]

    svc = SVC(kernel='linear', C=100)
    sv_svc = SVC(kernel='linear', C=100)

    svc.fit(xx, y)
    sv_xx = [xx[i] for i in svc.support_]
    sv_y = [y[i] for i in svc.support_]

    sv_svc.fit(sv_xx, sv_y)

    assert (abs(svc.score(xx, y) - sv_svc.score(xx, y)) < 0.01)
