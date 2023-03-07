import numpy as np

from fast_match import utils


class MatchConfig(object):
    def __init__(self, **kwargs):
        self.translate_x = kwargs.get("translate_x")
        self.translate_y = kwargs.get("translate_y")
        self.scale_x = kwargs.get("scale_x")
        self.scale_y = kwargs.get("scale_y")
        self.rotate_1 = kwargs.get("rotate_1")
        self.rotate_2 = kwargs.get("rotate_2")

        self.affine_matrix = self.generate_affine_matrix()

    def generate_affine_matrix(self):
        R1 = utils.generate_rotate_matrix(self.rotate_1)
        R2 = utils.generate_rotate_matrix(self.rotate_2)
        S = np.matrix([[self.scale_x, 0], [0, self.scale_y]])
        A = np.matrix([[0, 0, self.translate_x],
                       [0, 0, self.translate_y],
                       [0, 0, 1]])

        A[1:2, 1:2] = R1*S*R2

        return A

    def as_matrix(self):
        return np.matrix([self.translate_x, self.translate_y, self.rotate_2, self.scale_x, self.scale_y, self.rotate_1])

    def from_matrix(self, configs):
        result = []
        for i in range(configs.rows):
            result.append(configs[i][:5])

        return result
