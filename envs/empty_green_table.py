import sapien

from ._base_task import Base_Task
from .utils import create_box, create_table


class empty_green_table(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def create_table_and_wall(self, table_xy_bias=[0, 0], table_height=0.74):
        self.table_xy_bias = table_xy_bias
        table_height += self.table_z_bias
        self.wall_texture, self.table_texture = None, None

        wall_color = (0.85, 0.85, 0.85)
        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0, 1, 1.5]),
            half_size=[3, 0.6, 1.5],
            color=wall_color,
            name="wall",
            is_static=True,
        )

        self.table = create_table(
            self.scene,
            sapien.Pose(p=[table_xy_bias[0], table_xy_bias[1], table_height]),
            length=1.2,
            width=0.7,
            height=table_height,
            thickness=0.05,
            color=(0, 1, 0),
            name="table",
            is_static=True,
        )

    def load_actors(self):
        return

    def play_once(self):
        return self.info

    def check_success(self):
        return True
