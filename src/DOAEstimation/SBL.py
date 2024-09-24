#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import numpy as np

def SBL(raw_data, max_iteration=500, error_threshold=1e-3):
    """
    :param raw_data:
    :param max_iteration:
    :param error_threshold:
    :return:
    """
    A = np.exp(1j * np.pi * np.arange(num_sensors)[:, np.newaxis] * np.sin(np.deg2rad(angles)))
    mu = A.T.conjugate() @ np.linalg.pinv(A @ A.T.conjugate()) @ raw_data
    sigma2 = 0.1 * np.linalg.norm(raw_data, 'fro') ** 2 / (num_sensors * num_snapshots)
    gamma = np.diag((mu @ mu.T.conjugate()).real) / num_snapshots
    ItrIdx = 1
    stop_iter = False
    gamma0 = gamma
    while not stop_iter and ItrIdx < max_iteration:
        gamma0 = gamma
        Q = sigma2 * np.eye(self.num_sensors) + np.dot(np.dot(A, np.diag(gamma)), A.T.conjugate())
        Qinv = np.linalg.pinv(Q)
        Sigma = np.diag(gamma) - np.dot(np.dot(np.dot(np.diag(gamma), A.T.conjugate()), Qinv),
                                        np.dot(A, np.diag(gamma)))
        mu = np.dot(np.dot(np.diag(gamma), A.T.conjugate()), np.dot(Qinv, raw_data))
        sigma2 = ((np.linalg.norm(raw_data - np.dot(A, mu), 'fro') ** 2 + self.num_snapshots * np.trace(
            np.dot(np.dot(A, Sigma), A.T.conjugate()))) /
                  (self.num_sensors * self.num_snapshots)).real
        mu_norm = np.diag(mu @ mu.T.conjugate()) / self.num_snapshots
        gamma = np.abs(mu_norm + np.diag(Sigma))

        if np.linalg.norm(gamma - gamma0) / np.linalg.norm(gamma) < error_threshold:
            stop_iter = True
        ItrIdx += 1
    return gamma
