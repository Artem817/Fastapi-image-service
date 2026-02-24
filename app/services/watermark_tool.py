from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from pathlib import Path
import random
import math


class WatermarkEngine:
    def __init__(self, opacity=0.3, rotation=30, scale_percent=0.2, density=2, randomize=False, jitter=0.2, seed=None):
        """
        :param opacity: Watermark transparency (0.1 - 1.0)
        :param rotation: Angle of rotation
        :param scale_percent: Logo size relative to photo width (0.2 = 20%)
        :param density: Mesh density (the lower the number, the denser)
        :param randomize: Add randomized offsets to the watermark grid
        :param jitter: Jitter ratio relative to spacing (0.0 - 0.9)
        :param seed: Optional random seed for reproducible patterns
        """
        self.opacity = opacity
        self.rotation = rotation
        self.scale_percent = scale_percent
        self.density = density
        self.randomize = randomize
        self.jitter = jitter
        self.seed = seed

    def _prepare_logo(self, img_pil, logo_path):
        logo = Image.open(logo_path).convert("RGBA")
        r, g, b, a = logo.split()
        alpha_extrema = a.getextrema()
        alpha_is_flat = alpha_extrema[0] == alpha_extrema[1]
        if alpha_is_flat:
            logo_rgb = logo.convert("RGB")
            arr = np.array(logo_rgb, dtype=np.float32)
            border = np.concatenate(
                [arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]],
                axis=0,
            )
            bg = np.median(border, axis=0)
            dist = np.sqrt(np.sum((arr - bg) ** 2, axis=2))
            border_dist = np.sqrt(np.sum((border - bg) ** 2, axis=1))
            threshold = max(5.0, float(np.percentile(border_dist, 90)))

            if dist.max() > threshold + 1.0:
                mask = (dist - threshold) / (dist.max() - threshold)
                mask = np.clip(mask, 0.0, 1.0)
                mask = (mask ** 1.2) * 255.0
                a = Image.fromarray(mask.astype(np.uint8), mode="L")
                a = ImageOps.autocontrast(a)
            else:
                logo_gray = ImageOps.grayscale(logo_rgb)
                logo_gray = ImageOps.autocontrast(logo_gray)
                if np.mean(np.array(logo_gray)) > 127:
                    a = ImageOps.invert(logo_gray)
                else:
                    a = logo_gray
                a = ImageEnhance.Contrast(a).enhance(1.5)

        logo_colored = Image.merge("RGBA", (r, g, b, a))

        w, h = img_pil.size
        target_logo_w = max(1, int(min(w, h) * self.scale_percent))
        aspect_ratio = logo.height / logo.width
        target_logo_h = max(1, int(target_logo_w * aspect_ratio))

        logo_base_resized = logo.convert("RGB").resize((target_logo_w, target_logo_h), Image.LANCZOS)
        logo_resized = logo_colored.resize((target_logo_w, target_logo_h), Image.LANCZOS)

        r, g, b, a = logo_resized.split()
        alpha_floor = int(255 * min(self.opacity, 0.35)) if alpha_is_flat else 0

        def scale_alpha(p):
            if p <= 0:
                return 0
            scaled = int(p * self.opacity)
            if alpha_floor:
                return max(scaled, alpha_floor)
            return scaled

        a = a.point(scale_alpha)
        if float(np.mean(np.array(a))) < 2.0:
            fallback_mask = ImageOps.grayscale(logo_base_resized)
            fallback_mask = ImageOps.autocontrast(fallback_mask)
            if np.mean(np.array(fallback_mask)) > 127:
                fallback_mask = ImageOps.invert(fallback_mask)
            fallback_mask = fallback_mask.point(lambda p: int(p * self.opacity))
            if alpha_floor:
                fallback_mask = fallback_mask.point(lambda p: max(p, alpha_floor) if p > 0 else 0)
            a = fallback_mask
        logo_final = Image.merge("RGBA", (r, g, b, a))

        return logo_final

    def apply(self, img_path, logo_path, output_path=None):
        img = Image.open(img_path).convert("RGBA")
        logo = self._prepare_logo(img, logo_path)

        img_w, img_h = img.size
        logo_w, logo_h = logo.size

        diagonal = int(math.sqrt(img_w ** 2 + img_h ** 2) * 1.5)
        canvas = Image.new('RGBA', (diagonal, diagonal), (0, 0, 0, 0))

        spacing_x = max(1, int(logo_w * self.density))
        spacing_y = max(1, int(logo_h * self.density))

        rng = random.Random(self.seed) if self.randomize else None
        start_x = rng.randint(0, spacing_x - 1) if rng else 0
        start_y = rng.randint(0, spacing_y - 1) if rng else 0

        max_jitter_x = max(0, (spacing_x - logo_w) // 2)
        max_jitter_y = max(0, (spacing_y - logo_h) // 2)
        jitter_x = min(int(spacing_x * self.jitter), max_jitter_x)
        jitter_y = min(int(spacing_y * self.jitter), max_jitter_y)

        for y in range(-spacing_y, diagonal + spacing_y, spacing_y):
            for x in range(-spacing_x, diagonal + spacing_x, spacing_x):
                offset = (logo_w // 2) if (y // spacing_y) % 2 == 1 else 0
                pos_x = x + offset + start_x
                pos_y = y + start_y
                if rng and (jitter_x > 0 or jitter_y > 0):
                    pos_x += rng.randint(-jitter_x, jitter_x)
                    pos_y += rng.randint(-jitter_y, jitter_y)
                canvas.paste(logo, (pos_x, pos_y), logo)

        canvas_rotated = canvas.rotate(self.rotation, resample=Image.BICUBIC)

        c_w, c_h = canvas_rotated.size
        left = (c_w - img_w) // 2
        top = (c_h - img_h) // 2

        watermark_layer = canvas_rotated.crop((left, top, left + img_w, top + img_h))

        result = Image.alpha_composite(img, watermark_layer)

        result = result.convert("RGB")
        if output_path:
            result.save(output_path, quality=95)
            print(f"Saved: {output_path}")

        return result
