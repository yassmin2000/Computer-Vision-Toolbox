import numpy as np
from scipy.signal import convolve2d
from scipy.interpolate import RectBivariateSpline
from skimage import img_as_float
# from skimage.util.dtype import _supported_float_type
def active_contour( image, snake, alpha=0.01, beta=0.1,
                       w_line=0, w_edge=1, gamma=0.01,
                       max_px_move=1.0,
                       max_num_iter=2500, convergence=0.1):
        """
    Implements the active contour model for image segmentation.

    Args:
        image: Input image.
        snake: Initial snake contour.
        alpha: Weight of the continuity term.
        beta: Weight of the curvature term.
        w_line: Weight of the line integral term.
        w_edge: Weight of the edge integral term.
        gamma: Weight of the gradient magnitude term.
        max_px_move: Maximum pixel movement.
        max_num_iter: Maximum number of iterations.
        convergence: Convergence criterion.

    Returns:
        Contour points after segmentation.
    """
        max_num_iter = int(max_num_iter)
        if max_num_iter <= 0:
            raise ValueError("max_num_iter should be >0.")
        convergence_order = 10
        img = img_as_float(image)
        float_dtype = np.float64  # or np.float32

        img = img.astype(float_dtype, copy=False)

        RGB = img.ndim == 3

        # Precompute intensity image and edges
        if w_edge != 0:
            if RGB:
                edges = [convolve2d(img[..., i], np.array([[-1, 0, 1]]), mode='same') ** 2 +
                         convolve2d(img[..., i], np.array([[-1, 0, 1]]).T, mode='same') ** 2
                         for i in range(3)]
            else:
                edges = [convolve2d(img, np.array([[-1, 0, 1]]), mode='same') ** 2 +
                         convolve2d(img, np.array([[-1, 0, 1]]).T, mode='same') ** 2]
        else:
            edges = [0]

        if RGB:
            img = w_line * np.sum(img, axis=2) + w_edge * sum(edges)
        else:
            img = w_line * img + w_edge * edges[0]

        intp = RectBivariateSpline(np.arange(img.shape[1]),
                                    np.arange(img.shape[0]),
                                    img.T, kx=2, ky=2, s=0)

        snake_xy = snake[:, ::-1]
        x = snake_xy[:, 0].astype(float_dtype)
        y = snake_xy[:, 1].astype(float_dtype)
        n = len(x)
        xsave = np.empty((convergence_order, n), dtype=float_dtype)
        ysave = np.empty((convergence_order, n), dtype=float_dtype)

        eye_n = np.eye(n, dtype=float_dtype)
        a = (np.roll(eye_n, -1, axis=0) + np.roll(eye_n, -1, axis=1) - 2 * eye_n)
        b = (np.roll(eye_n, -2, axis=0) + np.roll(eye_n, -2, axis=1) - 4 * np.roll(eye_n, -1, axis=0) -
             4 * np.roll(eye_n, -1, axis=1) + 6 * eye_n)
        A = -alpha * a + beta * b

        inv = np.linalg.inv(A + gamma * eye_n).astype(float_dtype, copy=False)

        for _ in range(max_num_iter):
            fx = intp(x, y, dx=1, grid=False)
            fy = intp(x, y, dy=1, grid=False)

            xn = inv @ (gamma * x + fx)
            yn = inv @ (gamma * y + fy)
            for i in range(n):
                dx = max_px_move * np.tanh(xn - x)
                dy = max_px_move * np.tanh(yn - y)
                x[i] += dx[i]
                y[i] += dy[i]

            j = _ % (convergence_order + 1)
            if j < convergence_order:
                xsave[j, :] = x
                ysave[j, :] = y
            else:
                dist = np.min(np.max(np.abs(xsave - x[None, :]) + np.abs(ysave - y[None, :]), 1))
                if dist < convergence:
                    break

        return np.stack([y, x], axis=1)
