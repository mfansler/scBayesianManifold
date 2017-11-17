from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def plot_C_arrows(x, t, C, scale_C=1, alpha_C=0.5):
    plt.figure()
    C_fwd = scale_C * C

    ax = plt.axes(projection='3d')

    ax.quiver3D(x[0, :], x[1, :], x[2, :],
                C_fwd[0, :], C_fwd[1, :], C_fwd[2, :],
                alpha=alpha_C)

    # points colored by t value
    ax.scatter3D(x[0, :], x[1, :], x[2, :], c=t[0], marker='o')
    return ax


def plot_G(t, x, G, alpha_G=0.7, color_G='gray'):
    plt.figure()
    ax = plt.axes(projection='3d')

    # generate collection of all edges (assumes symmetric)
    segments = [x[:, [i, j]].T for (i, j) in G.todok().keys() if i < j]
    ax.add_collection(Line3DCollection(segments, colors=color_G, alpha=alpha_G))

    # plot points, colored by t value
    ax.scatter3D(x[0, :], x[1, :], x[2, :], c=t, marker='o')
    return ax
