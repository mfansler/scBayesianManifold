import numpy as np
import numpy.linalg as ln
from scipy.linalg import solve_triangular
from scipy.stats import ortho_group
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from networkx.convert_matrix import from_scipy_sparse_matrix as sp_to_nx_graph
from networkx.algorithms.shortest_paths.weighted import all_pairs_dijkstra_path_length as dijkstra_dists
from sklearn.neighbors import kneighbors_graph

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


def infer_graph(x, k=5, keep_asymmetries=True, delta=2):
    # retrieve distance graph
    D = kneighbors_graph(x.T, k, mode='distance')
    # remove edges greater than delta*std
    D.data = np.multiply(D.data, D.data < (D.data.mean() + delta*D.data.std()))
    D.eliminate_zeros()

    # Adjacency Matrix
    G = D.copy()
    G.data = np.ones_like(G.data)
    # symmetrize
    G = G + G.T
    G.data = np.heaviside(G.data - 1, keep_asymmetries)
    G.eliminate_zeros()

    # ToDo: Add code to reconnect multicomponents

    return G


def initialize_t(G, x):
    N = G.shape[0]
    D = G.multiply(squareform(pdist(x.T)))
    H = sp_to_nx_graph(D)

    max_dist = 0
    max_t = None

    # find the node furtherest from all others
    for (i, ds) in dijkstra_dists(H):
        cur_dist = ln.norm(list(ds.values()))
        if cur_dist > max_dist:
            max_dist = cur_dist
            max_t = ds

    # extract distances
    t = sorted([(k, v) for k, v in max_t.items()], key=lambda tp: tp[0])
    t = np.array([entry[1] for entry in t])

    # standardize
    t = (t - t.mean()) / t.std()

    return t.reshape((1, N))


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