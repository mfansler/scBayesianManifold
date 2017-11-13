import numpy as np
import numpy.linalg as ln
from scipy.linalg import solve_triangular
from scipy.stats import ortho_group


def matrix_normal(A, M, U, V):
    n, p = A.shape
    A = np.matrix(A); M = np.matrix(M); U = np.matrix(U); V = np.matrix(V)
    return np.exp(-.5 * np.trace(ln.inv(V) * (A-M).T * ln.inv(U) * (A-M))) / ((2 * np.pi)**(n*p/2.0) * ln.det(V)**(n/2.0) * ln.det(U)**(p/2.0))


def matrix_normal_log(A, M, U, V):
    n, p = A.shape
    A = np.matrix(A); M = np.matrix(M); U = np.matrix(U); V = np.matrix(V)
    return -.5 * np.trace(ln.solve(V, (A-M).T) * ln.solve(U, (A-M))) - np.log(2 * np.pi)*(n*p/2.0) - np.log(ln.det(V))*(n/2.0) - np.log(ln.det(U))*(p/2.0)


def matrix_normal_log_star(A, M, U, V):
    A = np.matrix(A); M = np.matrix(M); U = np.matrix(U); V = np.matrix(V)
    return -.5 * np.trace(ln.solve(V, (A-M).T) * ln.solve(U, (A-M)))


def matrix_normal_log_star2(A, M, U_inv, V_inv):
    R = A - M
    return -.5 * np.trace(V_inv.dot(R.T.dot(U_inv.dot(R))))


def matrix_normal_log_star_std(A, V_inv):
    return -.5 * np.trace(V_inv.dot(A.T).dot(A))


def chol_inv(X):
    L = ln.cholesky(X)
    Linv = solve_triangular(L, np.identity(L.shape[0]), lower=True)
    return Linv.T.dot(Linv)


def randspd(n):
    D = np.diag(np.arange(0.5, 2, 1.5 / n))  # eigenvalues
    Q = ortho_group.rvs(n)  # random rotation
    return ln.multi_dot([Q, D, Q.T])


if __name__ == '__main__':

    from timeit import timeit

    n = 200

    # generate random nxn SPD matrices
    M = randspd(n)

    # random vector
    v = np.random.rand(n).T

    # Test validity of chol_inv
    M_inv = chol_inv(M)
    assert np.allclose(M_inv.dot(v), ln.solve(M, v)), "chol_inv not valid"

    # Test
    A = randspd(n)
    U = randspd(n)
    V = randspd(n)
    U_inv, V_inv = chol_inv(U), chol_inv(V)

    llA1 = matrix_normal_log_star(A, M, U, V)
    llA2 = matrix_normal_log_star2(A, M, U_inv, V_inv)
    assert np.isclose(llA1, llA2), "ll(MN*) is not stable"

    #print(timeit("matrix_normal_log_star(A, M, U, V)", number=100, globals=globals()))
    #print(timeit("matrix_normal_log_star2(A, M, U, V)", number=100, globals=globals()))

    # Test matrix_normal_log_star_std
    stepsize = 0.01
    A2 = np.random.randn(3, n)
    M2 = np.zeros_like(A2)
    U_inv2 = np.eye(3)

    llA2_1 = matrix_normal_log_star2(A2, M2, U_inv2, V_inv)
    llA2_2 = matrix_normal_log_star_std(A2, V_inv)
    assert np.isclose(llA2_1, llA2_2)

    print("All tests passed!")