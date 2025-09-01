from dataclasses import dataclass, field
from io import StringIO
from typing import Optional, Tuple, Union
from PIL import Image

import numpy as np

from .color import Color

Number = Union[int, float, np.number]
ColorLike = Union[
    "Color", Tuple[Number, Number, Number], Tuple[Number, Number, Number, Number], np.ndarray
]


@dataclass
class Canvas:
    width: Union[int, np.uint16]
    height: Union[int, np.uint16]
    channels: int = 4
    dtype: np.dtype = np.float32
    pixels: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        self.width = int(self.width)
        self.height = int(self.height)
        if self.channels not in (1, 3, 4):
            raise ValueError("channels must be 1, 3, or 4")
        if self.pixels is None:
            self.pixels = np.zeros((self.height, self.width, self.channels), dtype=self.dtype)
        else:
            arr = np.asarray(self.pixels)
            if arr.shape != (self.height, self.width, self.channels):
                raise ValueError(
                    f"pixels shape must be (height, width, channels) = "
                    f"({self.height}, {self.width}, {self.channels}); got {arr.shape}"
                )
            if arr.dtype != self.dtype:
                self.pixels = arr.astype(self.dtype, copy=False)
            else:
                self.pixels = arr

    def _clamp_array(self, arr: np.ndarray) -> np.ndarray:
        if np.issubdtype(self.dtype, np.floating):
            return np.clip(arr, 0.0, 1.0).astype(self.dtype, copy=False)
        else:
            return np.clip(arr, 0, 255).astype(self.dtype, copy=False)

    def _normalize_colorlike(self, c: ColorLike) -> np.ndarray:
        """
        Return an array shaped (channels,) matching this canvas' channels and dtype.
        Accepts:
          - Color object with methods to_tuple() or to_uint8_tuple()
          - tuple/list length 3 or 4
          - numpy array length 3 or 4
        Normalizes to canvas dtype:
          - float: 0..1
          - uint8: 0..255
        """
        if isinstance(c, np.ndarray):
            arr = np.asarray(c).astype(self.dtype, copy=False)
        else:
            try:
                if hasattr(c, "to_tuple"):
                    arr = np.asarray(c.to_tuple()[: self.channels])
                elif hasattr(c, "to_uint8_tuple") and np.issubdtype(self.dtype, np.integer):
                    arr = np.asarray(c.to_uint8_tuple()[: self.channels])
                else:
                    arr = np.asarray(tuple(c)[: self.channels])
            except Exception:
                raise TypeError("Unsupported ColorLike type: " + repr(type(c)))
        if arr.shape[0] == 3 and self.channels == 4:
            alpha_val = 1.0 if np.issubdtype(self.dtype, np.floating) else 255
            arr = np.concatenate([arr, np.array([alpha_val])])
        if arr.shape[0] != self.channels:
            raise ValueError(f"Color-like must have {self.channels} components, got {arr.shape[0]}")
        if np.issubdtype(self.dtype, np.integer):
            if np.issubdtype(arr.dtype, np.floating):
                arr = (arr * 255.0).round()
            arr = arr.astype(self.dtype, copy=False)
            return self._clamp_array(arr)
        else:
            if np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(np.float32) / 255.0
            arr = arr.astype(self.dtype, copy=False)
            return self._clamp_array(arr)

    def _check_coords(self, x: int, y: int):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        return True

    def set_pixel(self, x: int, y: int, color: ColorLike):
        """Set pixel at integer coordinates (x, y). Origin (0,0) is top-left. Uses clamping."""
        if self._check_coords(x, y):
            arr = self._normalize_colorlike(color)
            self.pixels[y, x, :] = arr

    def get_pixel(self, x: int, y: int, as_color: bool = False):
        """Get pixel at (x, y). If as_color=True and a Color class exists, returns Color; otherwise numpy array."""
        if not self._check_coords(x, y):
            return

        px = self.pixels[y, x].copy()
        if as_color:
            try:
                if np.issubdtype(self.dtype, np.integer):
                    r, g, b, *rest = px.tolist()
                    a = rest[0] if rest else 255
                    return Color.from_uint8(int(r), int(g), int(b), int(a))
                else:
                    r, g, b, *rest = px.tolist()
                    a = rest[0] if rest else 1.0
                    return Color.from_floats(float(r), float(g), float(b), float(a))
            except Exception:
                return px
        return px

    def fill(self, color: ColorLike):
        """Fill entire canvas with color."""
        arr = self._normalize_colorlike(color)
        self.pixels[:, :, :] = arr

    def clear(self):
        """Set all pixels to zero (transparent black if 4 channels)."""
        if np.issubdtype(self.dtype, np.integer):
            self.pixels.fill(0)
        else:
            self.pixels.fill(0.0)

    def draw_hline(self, x0: int, x1: int, y: int, color: ColorLike):
        if x1 < x0:
            x0, x1 = x1, x0
        x0 = max(0, x0)
        x1 = min(self.width - 1, x1)
        if not (0 <= y < self.height):
            return
        arr = self._normalize_colorlike(color)
        self.pixels[y, x0 : x1 + 1, :] = arr

    def draw_vline(self, x: int, y0: int, y1: int, color: ColorLike):
        if y1 < y0:
            y0, y1 = y1, y0
        y0 = max(0, y0)
        y1 = min(self.height - 1, y1)
        if not (0 <= x < self.width):
            return
        arr = self._normalize_colorlike(color)
        self.pixels[y0 : y1 + 1, x, :] = arr

    def draw_rect(self, x: int, y: int, w: int, h: int, color: ColorLike, filled: bool = False):
        if w <= 0 or h <= 0:
            return
        x0 = x
        y0 = y
        x1 = x + w - 1
        y1 = y + h - 1
        if filled:
            x0c = max(0, x0)
            x1c = min(self.width - 1, x1)
            y0c = max(0, y0)
            y1c = min(self.height - 1, y1)
            if x1c < x0c or y1c < y0c:
                return
            arr = self._normalize_colorlike(color)
            self.pixels[y0c : y1c + 1, x0c : x1c + 1, :] = arr
        else:
            self.draw_hline(x0, x1, y0, color)
            self.draw_hline(x0, x1, y1, color)
            self.draw_vline(x0, y0, y1, color)
            self.draw_vline(x1, y0, y1, color)

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: ColorLike):
        """Bresenham line algorithm (integer coords)."""
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            if 0 <= x0 < self.width and 0 <= y0 < self.height:
                self.pixels[y0, x0, :] = self._normalize_colorlike(color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def blit(self, src: "Canvas", dest_x: int = 0, dest_y: int = 0, alpha: bool = False):
        """
        Copy src onto this canvas at (dest_x, dest_y).
        If alpha=True and both canvases have 4 channels, alpha compositing is applied (simple source-over).
        """
        if not isinstance(src, Canvas):
            raise TypeError("src must be a Canvas")
        sx0 = max(0, -dest_x)
        sy0 = max(0, -dest_y)
        sx1 = min(src.width, self.width - dest_x)
        sy1 = min(src.height, self.height - dest_y)
        if sx1 <= sx0 or sy1 <= sy0:
            return
        tx0 = dest_x + sx0
        ty0 = dest_y + sy0
        w = sx1 - sx0
        h = sy1 - sy0
        src_block = src.pixels[sy0 : sy0 + h, sx0 : sx0 + w, :]
        dst_block = self.pixels[ty0 : ty0 + h, tx0 : tx0 + w, :]

        if alpha and src.channels == 4 and self.channels == 4:
            if np.issubdtype(self.dtype, np.integer):
                src_f = src_block.astype(np.float32) / 255.0
                dst_f = dst_block.astype(np.float32) / 255.0
            else:
                src_f = src_block.astype(np.float32, copy=False)
                dst_f = dst_block.astype(np.float32, copy=False)
            sa = src_f[..., 3:4]
            out_a = sa + dst_f[..., 3:4] * (1.0 - sa)
            with np.errstate(invalid="ignore", divide="ignore"):
                out_rgb = src_f[..., :3] * sa + dst_f[..., :3] * dst_f[..., 3:4] * (1.0 - sa)
                mask = out_a > 0
                out_rgb[mask[..., 0]] = out_rgb[mask[..., 0]] / out_a[mask[..., 0]]
            out = np.concatenate([out_rgb, out_a], axis=-1)
            if np.issubdtype(self.dtype, np.integer):
                out = np.clip((out * 255.0).round(), 0, 255).astype(self.dtype)
            else:
                out = np.clip(out, 0.0, 1.0).astype(self.dtype)
            self.pixels[ty0 : ty0 + h, tx0 : tx0 + w, :] = out
        else:
            if src_block.dtype != self.pixels.dtype:
                src_block = src_block.astype(self.pixels.dtype, copy=False)
            self.pixels[ty0 : ty0 + h, tx0 : tx0 + w, :] = src_block

    def to_ppm(self):
        """
        Returns a PPM (image format) string buffer representation of the canvas.
        """

        header = f"P3\n{self.width} {self.height}\n255\n"
        buf = StringIO()

        buf.write(header)

        for y, x in np.ndindex(self.height, self.width):
            px = self.pixels[y, x]

            buf.write(f"{round(px[0] * 255.0)} {round(px[1] * 255.0)} {round(px[2] * 255.0)}\n")

        return buf

    def to_image(self):
        """Return a Pillow Image for viewing or saving. Requires pillow (PIL)."""
        if self.channels == 1:
            if np.issubdtype(self.dtype, np.integer):
                arr = self.pixels[:, :, 0]
            else:
                arr = (self.pixels[:, :, 0] * 255.0).round().astype(np.uint8)
            return Image.fromarray(arr, mode="L")
        if self.channels == 3:
            if np.issubdtype(self.dtype, np.integer):
                arr = self.pixels[:, :, :3].astype(np.uint8)
            else:
                arr = (self.pixels[:, :, :3] * 255.0).round().astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")
        if self.channels == 4:
            if np.issubdtype(self.dtype, np.integer):
                arr = self.pixels.astype(np.uint8)
            else:
                arr = (self.pixels * 255.0).round().astype(np.uint8)
            return Image.fromarray(arr, mode="RGBA")

    def save(self, path: str):
        """Save canvas to a file (PNG, etc.) using Pillow."""
        im = self.to_image()
        im.save(path)

    @classmethod
    def from_image(cls, pil_image, dtype: np.dtype = np.float32) -> "Canvas":
        """Create a Canvas from a Pillow Image object (or path if passed string)."""
        if isinstance(pil_image, str):
            pil_image = Image.open(pil_image).convert("RGBA")
        else:
            pil_image = pil_image.convert("RGBA")
        arr = np.asarray(pil_image)
        h, w = arr.shape[:2]
        canv = cls(w, h, channels=4, dtype=dtype)
        if np.issubdtype(dtype, np.integer):
            canv.pixels = arr.astype(dtype, copy=False)
        else:
            canv.pixels = (arr.astype(np.float32) / 255.0).astype(dtype, copy=False)
        return canv

    def to_uint8_array(self) -> np.ndarray:
        """Return HxWxC uint8 array (0..255)."""
        if np.issubdtype(self.pixels.dtype, np.integer):
            return self.pixels.astype(np.uint8, copy=False)
        else:
            return (np.clip(self.pixels, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    def copy(self) -> "Canvas":
        return Canvas(
            self.width,
            self.height,
            channels=self.channels,
            dtype=self.dtype,
            pixels=self.pixels.copy(),
        )
