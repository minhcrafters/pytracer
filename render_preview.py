import numpy as np
import pygame

from core.canvas import Canvas
from core.color import Color
from core.objects.camera import Camera
from core.scene import Scene


class RenderPreview:
    def __init__(self, width: int, height: int):
        pygame.init()

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("render preview")

        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()

        self.surface = pygame.Surface((width, height))
        self.font = pygame.font.SysFont("Ubuntu Mono", 16)

        self.running = False

    def render_with_preview(
        self, scene: Scene, camera: Camera, update_frequency: int = 1000
    ) -> Canvas:
        """
        Render the scene with real-time pygame preview.

        Args:
            scene: The scene to render
            camera: Camera settings
            update_frequency: How often to update the display (pixels)

        Returns:
            The final rendered canvas
        """
        canvas = Canvas(camera.hsize, camera.vsize)
        total_pixels = camera.hsize * camera.vsize
        pixel_count = 0

        scale_x = self.width / camera.hsize
        scale_y = self.height / camera.vsize

        self.running = True

        for y, x in np.ndindex(camera.vsize, camera.hsize):
            if not self.running:
                break

            self._handle_events()

            ray = camera.ray_from_pixel(x, y)
            color = scene.color_at(ray)
            canvas.set_pixel(x, y, color)

            pygame_color = self._color_to_pygame(color)
            pygame.draw.rect(
                self.surface,
                pygame_color,
                (int(x * scale_x), int(y * scale_y), int(scale_x), int(scale_y)),
            )

            pixel_count += 1

            if pixel_count % update_frequency == 0 or pixel_count == total_pixels:
                self._update_display(scene, pixel_count, total_pixels)

        if self.running:
            self._update_display(scene, pixel_count, total_pixels, final=True)

        return canvas

    def _color_to_pygame(self, color: Color) -> tuple[int, int, int]:
        """Convert ray tracer color to pygame RGB tuple."""
        return (
            max(0, min(255, round(color.r * 255))),
            max(0, min(255, round(color.g * 255))),
            max(0, min(255, round(color.b * 255))),
        )

    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def _update_display(
        self, scene: Scene, pixel_count: int, total_pixels: int, final: bool = False
    ):
        """Update the pygame display with progress info."""
        self.screen.fill((0, 0, 0))
        self.screen.blit(
            pygame.transform.smoothscale(self.surface, (self.width, self.height)), (0, 0)
        )

        progress_percent = (pixel_count / total_pixels) * 100
        progress_text = "Rendering: {:.1f}%".format(progress_percent)

        if final:
            progress_text = f"Rendering Complete! {progress_text}"

        text_surf = self.font.render(progress_text, True, (255, 255, 255))
        self.screen.blit(text_surf, (10, 10))

        controls_text = "Click X to cancel"
        controls_surf = self.font.render(controls_text, True, (200, 200, 200))
        self.screen.blit(controls_surf, (10, self.height - 30))

        pygame.display.flip()
        self.clock.tick(60)

    def cleanup(self):
        """Clean up pygame resources."""
        pygame.quit()

    def wait_for_close(self):
        """Wait for user to close the window."""
        self.running = True
        while self.running:
            self._handle_events()
            self.clock.tick(60)


def render_scene_with_preview(
    scene: Scene, camera: Camera, window_width: int = 800, window_height: int = 600
) -> Canvas:
    """
    Convenience function to render a scene with pygame preview.

    Args:
        scene: Scene to render
        camera: Camera configuration
        window_width: Width of the preview window
        window_height: Height of the preview window

    Returns:
        Rendered canvas
    """
    preview = RenderPreview(window_width, window_height)
    try:
        canvas = preview.render_with_preview(scene, camera, update_frequency=1)
        if preview.running:
            preview.wait_for_close()
        return canvas
    finally:
        preview.cleanup()
