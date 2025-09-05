import multiprocessing as mp
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import pygame

from core.cpu.canvas import Canvas
from core.cpu.color import Color
from core.cpu.camera import Camera
from core.cpu.scene import Scene


class RenderPreview:
    def __init__(self, width: int, height: int):
        pygame.init()

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("render preview")

        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()

        self.surface = pygame.Surface((width, height))
        self.border = pygame.Surface((width, height), pygame.SRCALPHA)
        self.font = pygame.font.SysFont("Ubuntu Mono", 16)

        self.running = False

    def render_with_preview(
        self, scene: Scene, camera: Camera, update_frequency: int = 1000
    ) -> Canvas:
        """
        Render the scene with real-time pygame preview using parallel processing.

        Args:
            scene: The scene to render
            camera: Camera settings
            update_frequency: How often to update the display (pixels)

        Returns:
            The final rendered canvas
        """
        canvas = Canvas(camera.hsize, camera.vsize)
        total_pixels = camera.hsize * camera.vsize

        scale_x = self.width / camera.hsize
        scale_y = self.height / camera.vsize

        self.running = True

        pixels = [(x, y) for y in range(camera.vsize) for x in range(camera.hsize)]

        processed_pixels = 0

        with mp.Pool() as pool:
            for i in range(0, len(pixels), update_frequency):
                if not self.running:
                    break

                self._handle_events()

                batch = pixels[i : i + update_frequency]

                batch_results = pool.starmap_async(
                    _render_pixel_worker, [(scene, camera, x, y) for x, y in batch]
                )

                batch_colors = batch_results.get()

                for (x, y), color in zip(batch, batch_colors):
                    canvas.set_pixel(x, y, color)

                    pygame_color = self._color_to_pygame(color)
                    pygame.draw.rect(
                        self.surface,
                        pygame_color,
                        (int(x * scale_x), int(y * scale_y), int(scale_x), int(scale_y)),
                    )

                    self.border.fill((0, 0, 0, 0))
                    pygame.draw.rect(
                        self.border,
                        (128, 128, 128, 255),
                        (
                            int(x * scale_x) - 1,
                            int(y * scale_y) - 1,
                            int(scale_x) + 1,
                            int(scale_y) + 1,
                        ),
                        width=1,
                    )

                processed_pixels += len(batch)
                self._update_display(processed_pixels, total_pixels)

        if self.running:
            self._update_display(processed_pixels, total_pixels, final=True)

        return canvas

    def _color_to_pygame(self, color: Color) -> tuple[int, int, int]:
        return (
            max(0, min(255, round(color.r * 255))),
            max(0, min(255, round(color.g * 255))),
            max(0, min(255, round(color.b * 255))),
        )

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

    def _update_display(self, pixel_count: int, total_pixels: int, final: bool = False):
        self.screen.fill((0, 0, 0))
        scaled_base = pygame.transform.smoothscale(self.surface, (self.width, self.height))
        scaled_border = pygame.transform.smoothscale(self.border, (self.width, self.height))
        composited = scaled_base.copy()
        composited.blit(scaled_border, (0, 0))

        self.screen.blit(composited, (0, 0))

        progress_percent = (pixel_count / total_pixels) * 100
        progress_text = "Rendering: {:.1f}%".format(progress_percent)

        if final:
            progress_text = f"Rendering Complete!"

        text_surf = self.font.render(progress_text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(topleft=(10, 10))

        bg_surf = pygame.Surface(text_rect.inflate(20, 12).size, pygame.SRCALPHA)
        bg_surf.fill((0, 0, 0, 128))

        bg_surf.blit(text_surf, (10, 6))
        self.screen.blit(bg_surf, text_rect.topleft)

        pygame.display.flip()
        self.clock.tick(240)

    def cleanup(self):
        pygame.quit()

    def wait_for_close(self):
        self.running = True
        while self.running:
            self._handle_events()
            self.clock.tick(240)


def _render_pixel_worker(scene: Scene, camera: Camera, x: int, y: int) -> Color:
    """
    Worker function for parallel pixel rendering.
    This function must be at module level to be picklable for multiprocessing.

    Args:
        scene: Scene to render
        camera: Camera settings
        x: Pixel x coordinate
        y: Pixel y coordinate

    Returns:
        Color for the pixel
    """
    ray = camera.ray_from_pixel(x, y)
    return scene.color_at(ray)


def render_scene(
    scene: Scene, camera: Camera, window_width: int = 800, window_height: int = 600
) -> Canvas:
    preview = RenderPreview(window_width, window_height)
    try:
        canvas = preview.render_with_preview(scene, camera, update_frequency=1)
        if preview.running:
            preview.wait_for_close()
        return canvas
    finally:
        preview.cleanup()
