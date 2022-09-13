import toppra as ta
from toppra.constraint import LinearConstraint, DiscretizationType
import numpy as np
from toppra.interpolator import AbstractGeometricPath


class JointAccelerationConstraintVarying(LinearConstraint):
    def __init__(self, alim, alim_func=None, discretization_scheme=DiscretizationType.Interpolation):
        super(JointAccelerationConstraintVarying, self).__init__()
        alim = np.array(alim, dtype=float)
        if np.isnan(alim).any():
            raise ValueError("Bad velocity given: %s" % alim)
        if len(alim.shape) == 1:
            self.alim = np.vstack((-np.array(alim), np.array(alim))).T
        else:
            self.alim = np.array(alim, dtype=float)

        self.alim_func = alim_func

        self.dof = self.alim.shape[0]
        self.set_discretization_type(discretization_scheme)

        assert self.alim.shape[1] == 2, "Wrong input shape."
        self._format_string = "    Acceleration limit: \n"
        for i in range(self.alim.shape[0]):
            self._format_string += "      J{:d}: {:}".format(i + 1, self.alim[i]) + "\n"
        self.identical = True

    def compute_constraint_params(
        self, path: AbstractGeometricPath, gridpoints: np.ndarray, *args, **kwargs
    ):
        if path.dof != self.dof:
            raise ValueError(
                "Wrong dimension: constraint dof ({:d}) not equal to path dof ({:d})".format(
                    self.dof, path.dof
                )
            )
        ps_vec = (path(gridpoints, order=1)).reshape((-1, path.dof))
        pss_vec = (path(gridpoints, order=2)).reshape((-1, path.dof))
        dof = path.dof
        F_single = np.zeros((dof * 2, dof))


        g_single = []
        accelerations = [self.alim_func(x) for x in gridpoints]
        # print(accelerations)

        for i in range(len(accelerations)):
            g_single.append(np.zeros(dof * 2))
            g_single[-1][:dof] = accelerations[i][:, 1]
            g_single[-1][dof:] = -accelerations[i][:, 0]

        # print(g_single)
        g_single = np.array(g_single)

        F_single[0:dof, :] = np.eye(dof)
        F_single[dof:, :] = -np.eye(dof)

        if self.discretization_type == DiscretizationType.Collocation:
            return (
                ps_vec,
                pss_vec,
                np.zeros_like(ps_vec),
                F_single,
                g_single,
                None,
                None,
            )
        else:
            raise NotImplementedError("Other form of discretization not supported!")
