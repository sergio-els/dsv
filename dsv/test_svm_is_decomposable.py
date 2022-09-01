import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC


def test_svm_is_decomposable():
    for i in range(0, 100):
        test_svm_is_decomposable_try()


def test_svm_is_decomposable_try():
    # create a sample set which is linearly separable
    sample = make_classification(n_samples=1000, class_sep=1, n_features=4, n_redundant=2, n_informative=2,
                                 n_clusters_per_class=1, flip_y=0)
    X = sample[0]
    y = sample[1]

    # get the support vectors, the alphas and betas
    svc = SVC(kernel='linear', C=100)
    svc.fit(X, y)
    support_vector_indexes = svc.support_
    alphas = svc.dual_coef_[0]
    alphas_normalised = alphas / np.linalg.norm(alphas)
    # b = svc.intercept_[0]

    # now assume we have only found the support vectors. can we find the alphas and betas again?
    sv_X = [X[i] for i in support_vector_indexes]
    sv_y = [y[i] for i in support_vector_indexes]
    sv_svc = SVC(kernel='linear', C=100)
    sv_svc.fit(sv_X, sv_y)

    sv_support_vector_indexes = sv_svc.support_
    sv_alphas = sv_svc.dual_coef_[0]
    sv_alphas_normalised = sv_alphas / np.linalg.norm(sv_alphas)
    # sv_b = sv_svc.intercept_[0]

    # print(len(support_vector_indexes), svc.score(X, y), sv_svc.score(X, y), svc.score(X, y) - sv_svc.score(X, y))
    assert (abs(svc.score(X, y) - sv_svc.score(X, y)) < 0.01)

    # # first, the number of support vectors need to be the same
    # assert len(sv_support_vector_indexes) == len(
    #     support_vector_indexes), f"Support vectors different {support_vector_indexes}, {sv_support_vector_indexes}"
    #
    # assert (sv_support_vector_indexes == range(0, len(support_vector_indexes))).all()
    # # then the alphas need to be almost the same
    # # print(np.linalg.norm(sv_alphas_normalised - alphas_normalised))
    #
    # if np.linalg.norm(alphas_normalised - sv_alphas_normalised) > 0.001:
    #     for i, (a, sva) in enumerate(zip(alphas_normalised, sv_alphas_normalised)):
    #         if abs(a - sva) > 0.01:
    #             print(i, a, sva, a - sva, sv_X[i], sv_y[i])
    # alphas_diff = np.linalg.norm(alphas_normalised - sv_alphas_normalised)
    #
    # if alphas_diff > 0.01:
    #     print(f"Alpha diff norm is big {alphas_diff}")
    # assert alphas_diff < 0.01
