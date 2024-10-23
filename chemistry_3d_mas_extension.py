# chemistry_3d_mas_extension.py

import asyncio
import os

import omni.ui as ui
from omni.isaac.examples.base_sample import BaseSampleExtension
from omni.isaac.examples.user_examples.chemistry_3d_mas import Chemistry3DMAS
from omni.isaac.ui.ui_utils import btn_builder

class Chemistry3DMASExtension(BaseSampleExtension):
    def on_startup(self, ext_id: str):
        super().on_startup(ext_id)
        super().start_extension(
            menu_name="Chemistry",
            submenu_name="",
            name="Chemistry 3D MAS",
            title="Chemistry 3D MAS",
            doc_link="",  # Provide a documentation link if available
            overview="This example demonstrates a chemistry MAS simulation in Isaac Sim.",
            sample=Chemistry3DMAS(),
            file_path=os.path.abspath(__file__),
            number_of_extra_frames=1,
        )
        self.task_ui_elements = {}
        frame = self.get_frame(index=0)
        self.build_task_controls_ui(frame)

    def _on_start_simulation_button_event(self):
        asyncio.ensure_future(self.sample.on_start_simulation_async())
        self.task_ui_elements["Start Simulation"].enabled = False

    def post_reset_button_event(self):
        self.task_ui_elements["Start Simulation"].enabled = True

    def post_load_button_event(self):
        self.task_ui_elements["Start Simulation"].enabled = True

    def post_clear_button_event(self):
        self.task_ui_elements["Start Simulation"].enabled = False

    def build_task_controls_ui(self, frame):
        with frame:
            with ui.VStack(spacing=5):
                frame.title = "Task Controls"
                frame.visible = True
                dict = {
                    "label": "Start Simulation",
                    "type": "button",
                    "text": "Start Simulation",
                    "tooltip": "Start the Chemistry MAS Simulation",
                    "on_clicked_fn": self._on_start_simulation_button_event,
                }
                self.task_ui_elements["Start Simulation"] = btn_builder(**dict)
                self.task_ui_elements["Start Simulation"].enabled = False
