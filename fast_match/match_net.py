import math

from fast_match import match_config, utils


class MatchNet(object):
    def __init__(self, **kwargs):
        self.bound_translate_x = [
            kwargs.get("min_translate_x"),
            kwargs.get("max_translate_x"),
        ]
        self.bound_translate_y = [
            kwargs.get("min_translate_y"),
            kwargs.get("max_translate_y"),
        ]
        self.bound_rotate = [kwargs.get("min_rotate"), kwargs.get("max_rotate")]
        self.bound_scale = [kwargs.get("min_scale"), kwargs.get("max_scale")]

        self.steps_translate_x = (
            kwargs.get("delta") * kwargs.get("width") / math.sqrt(2)
        )
        self.steps_translate_y = (
            kwargs.get("delta") * kwargs.get("height") / math.sqrt(2)
        )
        self.steps_rotate = kwargs.get("delta") * math.sqrt(2)
        self.steps_scale = kwargs.get("delta") / math.sqrt(2)

        self.tx_steps = self.generate_x_translation_steps()
        self.ty_steps = self.generate_y_translation_steps()
        self.r_steps = self.generate_rotation_steps()
        self.s_steps = self.generate_scale_steps()

    def generate_x_translation_steps(self):
        tx_steps = []
        for x in utils.range_float(
            self.bound_translate_x[0], self.bound_translate_x[1], self.steps_translate_x
        ):
            tx_steps.append(x)

        if (
            self.bound_translate_x[1] - tx_steps[len(tx_steps) - 1]
            > 0.5 * self.steps_translate_x
        ):
            tx_steps.append(tx_steps[len(tx_steps) - 1 - 1] + self.steps_translate_x)

        return tx_steps

    def generate_y_translation_steps(self):
        ty_steps = []
        for y in utils.range_float(
            self.bound_translate_y[0], self.bound_translate_y[1], self.steps_translate_y
        ):
            ty_steps.append(y)

        if (
            self.bound_translate_y[1] - ty_steps[len(ty_steps) - 1]
            > 0.5 * self.steps_translate_y
        ):
            ty_steps.append(ty_steps[len(ty_steps) - 1 - 1] + self.steps_translate_y)

        return ty_steps

    def generate_rotation_steps(self):
        r_steps = []
        for r in utils.range_float(
            self.bound_rotate[0], self.bound_rotate[1], self.steps_rotate
        ):
            r_steps.append(r)

        if self.bound_rotate[1] - r_steps[len(r_steps) - 1] > 0.5 * self.steps_rotate:
            r_steps.append(r_steps[len(r_steps) - 1 - 1] + self.steps_rotate)

        return r_steps

    def generate_scale_steps(self):
        s_steps = []
        for s in utils.range_float(
            self.bound_scale[0], self.bound_scale[1], self.steps_scale
        ):
            s_steps.append(s)

        if self.bound_scale[1] - s_steps[len(s_steps) - 1] > 0.5 * self.steps_scale:
            s_steps.append(s_steps[len(s_steps) - 1 - 1] + self.steps_scale)

        return s_steps

    def multi(self, factor):
        self.steps_translate_x *= factor
        self.steps_translate_y *= factor
        self.steps_rotate *= factor
        self.steps_scale *= factor

    def create_list_configs(self):
        tx_steps_len = len(self.tx_steps)
        ty_steps_len = len(self.ty_steps)
        s_steps_len = len(self.s_steps)
        r1_steps_len = len(self.r_steps)
        r2_steps_len = r1_steps_len

        if (
            math.fabs((self.bound_rotate[1] - self.bound_rotate[0]) - (2 * math.pi))
            < 0.1
        ):
            r2_steps_len = len(
                list(
                    filter(
                        lambda r: r < (-math.pi / 2 + self.steps_rotate / 2),
                        self.r_steps,
                    )
                )
            )

        configs = []

        for tx_index in range(tx_steps_len):
            tx = self.tx_steps[tx_index]

            for ty_index in range(ty_steps_len):
                ty = self.ty_steps[ty_index]

                for r1_index in range(r1_steps_len):
                    r1 = self.r_steps[r1_index]

                    for r2_index in range(r2_steps_len):
                        r2 = self.r_steps[r2_index]

                        for sx_index in range(s_steps_len):
                            sx = self.s_steps[sx_index]

                            for sy_index in range(s_steps_len):
                                sy = self.s_steps[sy_index]

                                configs.append(
                                    match_config.MatchConfig(
                                        translate_x=tx,
                                        translate_y=ty,
                                        scale_x=sx,
                                        scale_y=sy,
                                        rotate_1=r1,
                                        rotate_2=r2,
                                    )
                                )

        return configs
