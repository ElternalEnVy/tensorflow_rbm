import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation


def tick_params():
    """Tick params used in `plt.tick_params` or `im.axes.tick_params` to
    plot images without labels, borders etc..
    """
    return dict(axis='both', which='both',
                bottom='off', top='off', left='off', right='off',
                labelbottom='off', labelleft='off', labelright='off')

def im_plot(X, n_rows=10, n_cols=10, shape=None, title=None,
            title_params=None, imshow_params=None):
    """Plot batch of images `X` on a single graph."""

    title_params = title_params or {}
    title_params.setdefault('fontsize', 8)
    title_params.setdefault('y', 0.95)
    num = X.shape[0]
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*shape[1]/100., n_rows*shape[0]/100.), dpi=100)
    # fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10), dpi=100)
    for i in range(num):
        row = (i // n_cols)
        col = i % n_cols
        img = X[i].reshape(shape)
        if n_rows == 1:
            axs[i].imshow(img, cmap='gray', aspect='auto', norm=mpl.colors.Normalize())
            axs[i].axis('off')
        else:
            axs[row, col].imshow(img, cmap='gray', aspect='auto', norm=mpl.colors.Normalize())
            axs[row, col].axis('off')
    fig.subplots_adjust(wspace=0, hspace=0)

    # if title is not None:
    #     plt.suptitle(title, **title_params)


def im_reshape(X, n_width=10, n_height=10, shape=None, normalize=False):
    """Reshape batch of images `X` to a single grid-image

    Returns
    -------
    X_reshaped : (H, W, C) or (H, W) np.ndarray
        Where H = `n_height` * `shape`[0],
              W = `n_width` * `shape`[1],
              C = `shape`[2] if `shape`[2] > 1 (3 or 4)
    """
    # check params
    X = np.asarray(X)
    if shape is None:
        shape = X.shape[1:]

    # reshape `X`
    Y = X[:(n_width * n_height), ...].copy()
    if len(shape) == 2:
        shape = (shape[0], shape[1], 1)
    Y = Y.reshape(-1, *shape)
    Z = np.zeros((n_height * shape[0], n_width * shape[1], shape[2]), dtype=Y.dtype)

    for i in range(n_height):
        for j in range(n_width):
            ind_Y = n_height * i + j
            if ind_Y < len(Y):
                Y_i = Y[ind_Y, ...]
                if normalize:
                    Y_i -= Y_i.min()
                    Y_i /= max(Y_i.ptp(), 1e-5)
                    Y_i /= Y_i.max()
                Z[i * shape[0]:(i + 1) * shape[0],
                  j * shape[1]:(j + 1) * shape[1], ...] = Y_i
    if Z.shape[2] == 1:
        Z = Z[:, :, 0]

    return Z

def im_gif(matrices, im, fig, fname=None, title_func=None,
           title_params=None, anim_params=None, save_params=None):
    """Animate `matrices`.

    Parameters
    ----------
    matrices : [np.ndarray]
        list of matrices to animate

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    if title_func is None:
        title_func = lambda i: str(i)

    title_params = title_params or {}
    title_params.setdefault('fontsize', 18)

    anim_params = anim_params or {}
    anim_params.setdefault('interval', 250)
    anim_params.setdefault('blit', True)

    save_params = save_params or {}
    save_params.setdefault('dpi', 80)
    save_params.setdefault('writer', 'imagemagick')

    def init():
        im.set_array([[]])
        return im,

    def animate(i):
        im.set_array(matrices[i])
        title = title_func(i)
        im.axes.set_title(title, **title_params)
        return im,

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(matrices), **anim_params)
    if fname:
        anim.save(fname, **save_params)
    return anim

def plot_confusion_matrix(C, labels=None, labels_fontsize=13, **heatmap_params):
    # default params
    labels = labels or range(C.shape[0])
    annot_fontsize = 14
    xy_label_fontsize = 21

    # set default params where possible
    if not 'annot' in heatmap_params:
        heatmap_params['annot'] = True
    if not 'fmt' in heatmap_params:
        heatmap_params['fmt'] = 'd' if C.dtype is np.dtype('int') else '.3f'
    if not 'annot_kws' in heatmap_params:
        heatmap_params['annot_kws'] = {'size': annot_fontsize}
    elif not 'size' in heatmap_params['annot_kws']:
        heatmap_params['annot_kws']['size'] = annot_fontsize
    if not 'xticklabels' in heatmap_params:
        heatmap_params['xticklabels'] = labels
    if not 'yticklabels' in heatmap_params:
        heatmap_params['yticklabels'] = labels

    # plot the stuff
    with plt.rc_context(rc={'xtick.labelsize': labels_fontsize,
                            'ytick.labelsize': labels_fontsize}):
        ax = sns.heatmap(C, **heatmap_params)
        plt.xlabel('predicted', fontsize=xy_label_fontsize)
        plt.ylabel('actual', fontsize=xy_label_fontsize)
        return ax

