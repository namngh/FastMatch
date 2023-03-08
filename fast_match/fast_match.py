import cv2
import numpy as np
import math

from fast_match import (
    match_net,
    match_config,
    utils,
    DEFAULT_DELTA,
    DEFAULT_MAX_SCALE,
    DEFAULT_EPSILON,
    DEFAULT_MIN_SCALE,
    DEFAULT_PHOTOMETRIC_INVARIANCE,
)


class FastMatch(object):
    def __init__(self, **kwargs):
        self.epsilon = kwargs.get("epsilon", DEFAULT_EPSILON)
        self.delta = kwargs.get("delta", DEFAULT_DELTA)
        self.photometric_invariance = kwargs.get(
            "photometric_invariance", DEFAULT_PHOTOMETRIC_INVARIANCE
        )
        self.min_scale = kwargs.get("min_scale", DEFAULT_MIN_SCALE)
        self.max_scale = kwargs.get("max_scale", DEFAULT_MAX_SCALE)

    def run(self, image, template):
        image = utils.preprocess_image(image)
        template = utils.preprocess_image(template)

        template_y, template_x = template.shape
        image_y, image_x = image.shape

        r1x = 0.5 * (template_x - 1)
        r1y = 0.5 * (template_y - 1)
        r2x = 0.5 * (image_x - 1)
        r2y = 0.5 * (image_y - 1)

        min_translate_x = -(r2x - r1x * self.min_scale)
        max_translate_x = -min_translate_x
        min_translate_y = -(r2y - r1y * self.min_scale)
        max_translate_y = -min_translate_y
        min_rotate = -math.pi
        max_rotate = math.pi

        net = match_net.MatchNet(
            min_translate_x=min_translate_x,
            max_translate_x=max_translate_x,
            min_translate_y=min_translate_y,
            max_translate_y=max_translate_y,
            min_rotate=min_rotate,
            max_rotate=max_rotate,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
            delta=self.delta,
            width=template_x,
            height=template_y,
        )

        image = cv2.GaussianBlur(image, (3, 3), 0)
        template = cv2.GaussianBlur(template, (3, 3), 0)

        no_of_points = round(10 / (self.epsilon**2))

        xs = np.random.uniform(1, template_x, (1, no_of_points)).astype("uint")
        ys = np.random.uniform(1, template_y, (1, no_of_points)).astype("uint")

        best_distances = []
        level = 0
        new_delta = self.delta
        delta_fact = 1.511

        while True:
            level += 1

            affines, configs = self.config_to_affine(net, image, template)
            configs_len = len(configs)

            distances = self.evaluate_configs(
                image, template, affines, xs, ys, self.photometric_invariance
            )

            best_distance = min(distances)
            best_distances.append(best_distance)

            min_index = distances.index(best_distance)
            best_config = configs[min_index]
            best_affine = best_config.affine_matrix
            print(best_distance)

            if (
                (best_distance < 0.005)
                or ((level > 2) and (best_distance < 0.015))
                or level >= 20
            ):
                break

            if level > 3:
                mean_value = sum(best_distances[level - 3 : level - 1]) / len(distances)

                if best_distance > mean_value * 0.97:
                    break

            (
                good_configs,
                thresh,
                too_high_percentage,
            ) = self.get_good_config_by_distance(
                configs, best_distance, new_delta, distances
            )

            if (
                too_high_percentage
                and (best_distance > 0.05)
                and ((level == 1) and (configs_len < 7.5e6))
            ) or ((best_distance > 0.1) and ((level == 1) and (configs_len < 5e6))):
                factor = 0.9
                new_delta = new_delta * factor
                level = 0
                net.multi(factor)
                configs = net.create_list_configs()

            else:
                new_delta = new_delta / delta_fact

                expanded_configs = self.random_expand_configs(
                    good_configs, net, level, 80, delta_fact
                )

                configs = good_configs + expanded_configs

            xs = np.random.uniform(1, template_x, (1, no_of_points)).astype("uint")
            ys = np.random.uniform(1, template_y, (1, no_of_points)).astype("uint")

        return self.calc_corners(image, template, best_affine)

    def config_to_affine(self, net, image, template):
        image_y, image_x = image.shape
        template_y, template_x = template.shape

        top_left = [-10, -10]
        bottom_right = [image_y + 10, image_x + 10]

        configs = net.create_list_configs()

        r1x = 0.5 * (template_x - 1)
        r1y = 0.5 * (template_y - 1)
        r2x = 0.5 * (image_x - 1)
        r2y = 0.5 * (image_y - 1)

        corners = np.matrix(
            [
                [
                    1 - (r1x + 1),
                    template_x - (r1x + 1),
                    template_x - (r1x + 1),
                    1 - (r1x + 1),
                ],
                [
                    1 - (r1y + 1),
                    1 - (r1y + 1),
                    template_y - (r1y + 1),
                    template_y - (r1y + 1),
                ],
                [1, 1, 1, 1],
            ]
        )
        translate = np.matrix(
            [
                [r2x + 1, r2y + 1],
                [r2x + 1, r2y + 1],
                [r2x + 1, r2y + 1],
                [r2x + 1, r2y + 1],
            ]
        )

        affines = []
        temp_configs = []
        for i in range(len(configs)):
            affine_corners = (
                configs[i].affine_matrix * corners
            ).transpose() + translate

            if (
                utils.validate_affine_corner(affine_corners[0], top_left, bottom_right)
                and utils.validate_affine_corner(
                    affine_corners[1], top_left, bottom_right
                )
                and utils.validate_affine_corner(
                    affine_corners[2], top_left, bottom_right
                )
                and utils.validate_affine_corner(
                    affine_corners[3], top_left, bottom_right
                )
            ):
                affines.append(configs[i].affine_matrix)
                temp_configs.append(configs[i])

        return (affines, temp_configs)

    def evaluate_configs(
        self, image, template, affines, xs, ys, photometric_invariance
    ):
        image_y, image_x = image.shape
        template_y, template_x = template.shape
        _, no_of_points = xs.shape

        r1x = 0.5 * (template_x - 1)
        r1y = 0.5 * (template_y - 1)
        r2x = 0.5 * (image_x - 1)
        r2y = 0.5 * (image_y - 1)

        no_of_configs = len(affines)

        padded = cv2.copyMakeBorder(image, image_y, image_y, 0, 0, cv2.BORDER_CONSTANT)

        vals_i1 = []
        for i in range(no_of_points):
            vals_i1.append(template[int(ys[0, i] - 1), int(xs[0, i] - 1)])

        xs_centered = xs.copy() - (r1x + 1)
        ys_centered = ys.copy() - (r1y + 1)

        distances = []
        for i in range(no_of_configs):
            a11 = affines[i][0, 0]
            a12 = affines[i][0, 1]
            a13 = affines[i][0, 2]
            a21 = affines[i][1, 0]
            a22 = affines[i][1, 1]
            a23 = affines[i][1, 2]

            tmp_1 = (r2x + 1) + a13 + 0.5
            tmp_2 = (r2y + 1) + a23 + 0.5 + image_y
            score = 0

            if not photometric_invariance:
                for j in range(no_of_points):
                    target_x = int(
                        a11 * xs_centered[0, j] + a12 * ys_centered[0, j] + tmp_1
                    )
                    target_y = int(
                        a21 * ys_centered[0, j] + a22 * ys_centered[0, j] + tmp_2
                    )

                    score += math.fabs(vals_i1[j] - padded[target_y - 1, target_x - 1])

            else:
                xs_target = []
                ys_target = []

                sum_x = 0
                sum_y = 0
                sum_x_squared = 0
                sum_y_squared = 0

                for j in range(no_of_points):
                    target_x = int(
                        a11 * xs_centered[0, j] + a12 * ys_centered[0, j] + tmp_1
                    )
                    target_y = int(
                        a21 * ys_centered[0, j] + a22 * ys_centered[0, j] + tmp_2
                    )

                    xi = vals_i1[j]
                    yi = padded[target_y - 1, target_x - 1]

                    xs_target[j] = xi
                    ys_target[j] = yi

                    sum_x += xi
                    sum_y += yi

                    sum_x_squared += xi * xi
                    sum_y_squared += yi * yi

                epsilon = 1e-7
                mean_x = sum_x / no_of_points
                mean_y = sum_y / no_of_points
                sigma_x = (
                    math.sqrt(
                        (sum_x_squared - (sum_x * sum_x) / no_of_points) / no_of_points
                    )
                    + epsilon
                )
                sigma_y = (
                    math.sqrt(
                        (sum_y_squared - (sum_y * sum_y) / no_of_points) / no_of_points
                    )
                    + epsilon
                )

                sigma_div = sigma_x / sigma_y
                temp = -mean_x + sigma_div * mean_y

                for j in range(no_of_points):
                    score += math.fabs(xs_target[j] - sigma_div * ys_target[j] + temp)

            distances.append(score / no_of_points)

        return distances

    def get_threshold_per_delta(self, delta):
        return 0.1341 * delta + 0.0278 - 0.002

    def get_threshold_configs(self, configs, distances, thresh):
        good_configs = []
        for i in range(len(distances)):
            if distances[i] <= thresh:
                good_configs.append(configs[i])

        return good_configs

    def get_good_config_by_distance(self, configs, best_distance, new_delta, distances):
        thresh = best_distance + self.get_threshold_per_delta(new_delta)

        good_configs = self.get_threshold_configs(configs, distances, thresh)
        no_of_configs = len(good_configs)

        while no_of_configs > 27000:
            thresh *= 0.99
            good_configs = self.get_threshold_configs(configs, distances, thresh)

            no_of_configs = len(good_configs)

        percentage = 1.0 * no_of_configs / len(configs)
        too_high_percentage = percentage > 0.022

        return (good_configs, thresh, too_high_percentage)

    def random_expand_configs(configs, net, level, no_of_points, delta_factor):
        factor = delta_factor**level

        half_step_tx = (net.step_translate_x / factor,)
        half_step_ty = (net.step_translate_y / factor,)
        half_step_r = (net.step_rotate / factor,)
        half_step_s = net.step_scale / factor

        random_vec = np.random.normal(0, 0.5, (1, no_of_points * len(configs)))

        configs_matrix = []
        for i in range(len(configs)):
            configs_matrix.append(configs[i].as_matrix())

        expanded = configs_matrix.copy()
        expanded = cv2.repeat(expanded, no_of_points, 1)

        ranges = np.matrix(
            [
                half_step_tx,
                half_step_ty,
                half_step_r,
                half_step_s,
                half_step_s,
                half_step_r,
            ]
        )
        ranges = cv2.repeat(ranges, no_of_points * len(configs), 1)

        expanded_configs = expanded + random_vec * ranges

        return match_config.MatchConfig.from_matrix(expanded_configs)

    def calc_corners(image, template, affine):
        image_y, image_x = image.shape
        template_y, template_x = template.shape

        r1x = 0.5 * (template_x - 1)
        r1y = 0.5 * (template_y - 1)
        r2x = 0.5 * (image_x - 1)
        r2y = 0.5 * (image_y - 1)

        a11 = affine[0, 0]
        a12 = affine[0, 1]
        a13 = affine[0, 2]
        a21 = affine[1, 0]
        a22 = affine[1, 1]
        a23 = affine[1, 2]

        c1x = a11 * (1 - (r1x + 1)) + a12 * (1 - (r1y + 1)) + (r2x + 1) + a13
        c1y = a21 * (1 - (r1x + 1)) + a22 * (1 - (r1y + 1)) + (r2y + 1) + a23

        c2x = a11 * (template_x - (r1x + 1)) + a12 * (1 - (r1y + 1)) + (r2x + 1) + a13
        c2y = a21 * (template_x - (r1x + 1)) + a22 * (1 - (r1y + 1)) + (r2y + 1) + a23

        c3x = (
            a11 * (template_x - (r1x + 1))
            + a12 * (template_y - (r1y + 1))
            + (r2x + 1)
            + a13
        )
        c3y = (
            a21 * (template_x - (r1x + 1))
            + a22 * (template_y - (r1y + 1))
            + (r2y + 1)
            + a23
        )

        c4x = a11 * (1 - (r1x + 1)) + a12 * (template_y - (r1y + 1)) + (r2x + 1) + a13
        c4y = a21 * (1 - (r1x + 1)) + a22 * (template_y - (r1y + 1)) + (r2y + 1) + a23

        return np.matrix([[c1x, c1y], [c2x, c2y], [c3x, c3y], [c4x, c4y]])
