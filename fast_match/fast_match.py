import cv2
import numpy as np
import math

from fast_match import match_net, utils


class FastMatch(object):
    def __init__(self, **kwargs):
        self.epsilon = kwargs.get("epsilon")
        self.delta = kwargs.get("delta")
        self.photometric_invariance = kwargs.get("photometric_invariance")
        self.min_scale = kwargs.get("min_scale")
        self.max_scale = kwargs.get("max_scale")

    def run(self, image, template):
        image = utils.preprocess_image(image)
        template = utils.preprocess_image(template)

        r1x = 0.5 * (template.cols - 1)
        r1y = 0.5 * (template.rows - 1)
        r2x = 0.5 * (image.cols - 1)
        r2y = 0.5 * (image.rows - 1)

        min_translate_x = -(r2x - r1x * self.min_scale)
        max_translate_x = -min_translate_x
        min_translate_y = -(r2y - r1y * self.min_scale)
        max_translate_y = -min_translate_y
        min_rotate = -math.pi
        max_rotate = math.pi

        match_net = match_net.MatchNet(min_translate_x=min_translate_x,
                                       max_translate_x=max_translate_x,
                                       min_translate_y=min_translate_y,
                                       max_translate_y=max_translate_y,
                                       min_rotate=min_rotate,
                                       max_rotate=max_rotate,
                                       delta=self.delta,
                                       width=template.cols,
                                       height=template.rows)

        image_blur = cv2.GaussianBlur(image, (2, 2), 0)
        template_blur = cv2.GaussianBlur(template, (2, 2), 0)

        no_of_points = math.round(10 / (self.epsilon ** 2))

        level = 0
        while (True):
            level += 1

            affines, temp_configs = self.config_to_affine(
                match_net, image, template)

    def config_to_affine(self, match_net, image, template):
        top_left = [-10, -10]
        bottom_right = [image.cols + 10, image.rows + 10]

        configs = match_net.create_list_configs()

        r1x = 0.5 * (template.cols - 1)
        r1y = 0.5 * (template.rows - 1)
        r2x = 0.5 * (image.cols - 1)
        r2y = 0.5 * (image.rows - 1)

        corners = np.matrix([[1 - (r1x + 1), template.cols - (r1x + 1), template.cols - (r1x + 1),  1 - (r1x + 1)],
                             [1 - (r1y + 1), 1 - (r1y + 1), template.rows -
                              (r1y + 1), template.rows - (r1y + 1)],
                             [1, 1, 1, 1]])
        translate = np.matrix(
            [[r2x + 1, r2y + 1], [r2x + 1, r2y + 1], [r2x + 1, r2y + 1], r2x + 1, r2y + 1])

        affines = []
        temp_configs = []
        for i in range(len(configs)):
            affine_corners = (
                configs[i].affine_matrix * corners).transpose() + translate

            if utils.validate_affine_corner(affine_corners[0], top_left, bottom_right) and utils.validate_affine_corner(affine_corners[1], top_left, bottom_right) and utils.validate_affine_corner(affine_corners[2], top_left, bottom_right) and utils.validate_affine_corner(affine_corners[3], top_left, bottom_right):
                affines.append(configs[i].affine_matrix)
                temp_configs.append(configs[i])

        return (affines, temp_configs)
