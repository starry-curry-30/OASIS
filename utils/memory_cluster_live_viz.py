"""
Live single-window viz: camera + status panel (per-frame cluster match, scores, rep update) + thumbnails.
Optional fly-in chip to cluster tile. Optional video capture.
"""

from __future__ import annotations

import colorsys
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _has_gui_display() -> bool:
    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _cluster_color_bgr(cluster_id: int) -> Tuple[int, int, int]:
    h = (cluster_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.82, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


class MemoryClusterLiveVisualizer:
    # Must match largest dx in _cluster_cell_with_backing (cell width = thumb_h + this).
    _THUMB_BACKING_MAX_DX = 10

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.update_stride = max(1, int(cfg.get("update_stride", 1)))
        self.window_name = str(cfg.get("window_name", "OASIS MemoryCluster"))
        self.panel_h = int(cfg.get("panel_height", 400))
        self.status_panel_w = int(
            cfg.get("status_panel_width", cfg.get("topdown_size", 520))
        )
        self.thumb_h = int(cfg.get("thumb_height", 110))
        # 0 = no cap (fill strip width); else upper bound on clusters per row.
        self.max_thumbs_per_row = int(cfg.get("max_thumbs_per_row", 0))
        self._fly_enabled = bool(cfg.get("fly_animation", True))
        self._fly_duration = max(0.05, float(cfg.get("fly_duration_sec", 0.48)))
        self._fly_chip_size = max(32, int(cfg.get("fly_chip_size", 92)))
        self._draw_yolo_boxes = bool(cfg.get("draw_yolo_boxes", True))
        # Same semantics as memory_cluster.max_observation_depth_m (meters); <=0 disables.
        self._max_observation_depth_m = float(cfg.get("max_observation_depth_m", 0.0) or 0.0)
        self._save_video = bool(cfg.get("save_video", False))
        self._video_path = str(cfg.get("video_path", "") or "").strip()
        self._video_fps = max(1.0, float(cfg.get("video_fps", 30.0)))
        fc = str(cfg.get("video_fourcc", "mp4v") or "mp4v").strip()
        self._video_fourcc = (fc + "    ")[:4]
        has_disp = _has_gui_display()
        self._show_gui = bool(cfg.get("show_window", True)) and has_disp
        self._video_failed = False
        # Lock size to first frame; composite height changes when clusters grow — variable size corrupts video.
        self._video_out_size: Optional[Tuple[int, int]] = None

        self._stride = 0
        self._lock = threading.Lock()
        self._composite_base: Optional[np.ndarray] = None
        self._fly_state: Dict[str, Any] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

        if not self.enabled:
            return
        if not self._show_gui and not self._save_video:
            logger.info(
                "[MemoryClusterLiveViz] No GUI display and save_video=false; skipping"
            )
            self.enabled = False
            return
        if self._save_video and not self._video_path:
            logger.warning(
                "[MemoryClusterLiveViz] save_video=true but video_path empty; disabling video"
            )
            self._save_video = False
        if self._save_video and not self._show_gui:
            logger.info(
                "[MemoryClusterLiveViz] Headless: recording only to %s",
                self._video_path,
            )

        self._running = True
        self._thread = threading.Thread(target=self._gui_loop, daemon=True)
        self._thread.start()
        if self._show_gui:
            logger.info("[MemoryClusterLiveViz] Started window: %s", self.window_name)

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _normalize_frame_to_video_size(self, img: np.ndarray) -> np.ndarray:
        frame = img
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        h, w = int(frame.shape[0]), int(frame.shape[1])
        if self._video_out_size is None:
            w2 = w + (w % 2)
            h2 = h + (h % 2)
            self._video_out_size = (w2, h2)
            if (w, h) != (w2, h2):
                frame = cv2.copyMakeBorder(
                    frame,
                    0,
                    h2 - h,
                    0,
                    w2 - w,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
            return np.ascontiguousarray(frame)
        ow, oh = self._video_out_size
        if (w, h) == (ow, oh):
            return np.ascontiguousarray(frame)
        return np.ascontiguousarray(
            cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
        )

    def _try_create_video_writer(self, width: int, height: int) -> Optional[cv2.VideoWriter]:
        path = self._video_path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        ext = os.path.splitext(path)[1].lower()
        user_fcc = (self._video_fourcc or "MJPG")[:4].ljust(4)

        if ext == ".mp4":
            candidates = [user_fcc, "avc1", "mp4v", "X264"]
        elif ext in (".mkv", ".webm"):
            candidates = [user_fcc, "MJPG", "XVID"]
        else:
            candidates = [user_fcc, "MJPG", "XVID", "mp4v"]

        seen = set()
        ordered: List[str] = []
        for c in candidates:
            c4 = (c.strip() + "    ")[:4]
            if c4 not in seen:
                seen.add(c4)
                ordered.append(c4)

        size = (width, height)
        for fcc4 in ordered:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fcc4)
            except Exception:
                continue
            wr = cv2.VideoWriter(path, fourcc, self._video_fps, size)
            if wr.isOpened():
                logger.info(
                    "[MemoryClusterLiveViz] Recording %s @ %.1f fps fourcc=%r size=%s",
                    path,
                    self._video_fps,
                    fcc4,
                    size,
                )
                return wr

        if ext == ".mp4":
            alt = os.path.splitext(path)[0] + ".avi"
            logger.warning(
                "[MemoryClusterLiveViz] MP4 open failed; fallback MJPG AVI: %s",
                alt,
            )
            self._video_path = alt
            for fcc4 in ("MJPG", "XVID"):
                fourcc = cv2.VideoWriter_fourcc(*fcc4)
                wr = cv2.VideoWriter(alt, fourcc, self._video_fps, size)
                if wr.isOpened():
                    logger.info(
                        "[MemoryClusterLiveViz] Recording %s @ %.1f fps fourcc=%r",
                        alt,
                        self._video_fps,
                        fcc4,
                    )
                    return wr

        logger.error(
            "[MemoryClusterLiveViz] VideoWriter failed for all codecs (path=%s)",
            path,
        )
        return None

    def _write_video_frame(
        self, writer: Optional[cv2.VideoWriter], img: np.ndarray
    ) -> Optional[cv2.VideoWriter]:
        if not self._save_video or self._video_failed or img is None or img.size == 0:
            return writer
        frame = self._normalize_frame_to_video_size(img)
        h, w = int(frame.shape[0]), int(frame.shape[1])
        if writer is None:
            writer = self._try_create_video_writer(w, h)
            if writer is None:
                self._video_failed = True
                return None
        if writer is not None and writer.isOpened():
            writer.write(frame)
        return writer

    def _gui_loop(self) -> None:
        if self._show_gui:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            except Exception as exc:
                logger.warning("[MemoryClusterLiveViz] Failed to create window: %s", exc)
                self._show_gui = False
                if not self._save_video:
                    self._running = False
                    return

        writer: Optional[cv2.VideoWriter] = None
        frame_interval = max(0.001, 1.0 / self._video_fps)

        while self._running:
            now = time.monotonic()
            with self._lock:
                base = None if self._composite_base is None else self._composite_base.copy()
                fly = dict(self._fly_state) if self._fly_state else None
            img = self._compose_display(base, fly, now)
            if img is not None:
                if self._save_video:
                    writer = self._write_video_frame(writer, img)
                if self._show_gui:
                    try:
                        cv2.imshow(self.window_name, img)
                    except Exception:
                        pass
            if self._show_gui:
                try:
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord("q"):
                        self._running = False
                except Exception:
                    pass
            elif self._save_video:
                time.sleep(frame_interval)
            else:
                time.sleep(0.03)

        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
            logger.info("[MemoryClusterLiveViz] Video writer closed: %s", self._video_path)
        if self._show_gui:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass

    def update(
        self,
        manager: Any,
        result: Optional[Dict[str, Any]],
        color_rgb: np.ndarray,
        frame_idx: int,
        observations: Optional[List[Any]] = None,
        viz_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled or manager is None:
            return
        if result is None:
            return
        self._stride += 1
        if self._stride % self.update_stride != 0:
            return

        try:
            composite, layout = self._render(
                manager,
                result,
                color_rgb,
                frame_idx,
                observations=observations,
                viz_context=viz_context,
            )
        except Exception as exc:
            logger.debug("[MemoryClusterLiveViz] Render failed: %s", exc)
            return

        with self._lock:
            self._composite_base = composite
            if (
                self._fly_enabled
                and result is not None
                and layout.get("has_thumb_strip")
            ):
                self._arm_fly_animation(
                    manager, int(result.get("cluster_id", -1)), color_rgb, layout
                )

    def _arm_fly_animation(
        self,
        manager: Any,
        cluster_id: int,
        color_rgb: np.ndarray,
        layout: Dict[str, Any],
    ) -> None:
        fm = layout.get("thumb_stack")
        h1 = int(layout["h1"])
        gap = int(layout["gap"])
        Wc = int(layout["Wc"])
        sx, sy = Wc // 2, h1 // 2
        th = self.thumb_h
        if fm and cluster_id in fm.get("order", []):
            order = list(fm["order"])
            idx = order.index(int(cluster_id))
            pad = int(fm.get("pad", 0))
            cols = max(1, int(fm.get("cols", 1)))
            cell_w = int(fm["cell_w"])
            cell_h = int(fm["cell_h"])
            row = idx // cols
            col = idx % cols
            fx = float(fm.get("fly_scale_x", 1.0))
            ex = int((pad + col * cell_w + th // 2) * fx)
            ey = h1 + gap + pad + row * cell_h + th // 2
        else:
            ex = th // 2
            ey = h1 + gap + th // 2
        chip_rgb = self._resize_rgb_panel(color_rgb, self._fly_chip_size)
        chip_bgr = cv2.cvtColor(chip_rgb, cv2.COLOR_RGB2BGR)
        ring = _cluster_color_bgr(cluster_id)
        self._fly_state = {
            "t0": time.monotonic(),
            "duration": self._fly_duration,
            "sx": sx,
            "sy": sy,
            "ex": ex,
            "ey": ey,
            "chip": chip_bgr.copy(),
            "ring": ring,
        }

    def _compose_display(
        self,
        base: Optional[np.ndarray],
        fly: Optional[Dict[str, Any]],
        now: float,
    ) -> Optional[np.ndarray]:
        if base is None:
            return None
        if not fly or fly.get("chip") is None:
            return base
        t0 = float(fly["t0"])
        duration = float(fly["duration"])
        t = (now - t0) / duration
        if t >= 1.0:
            with self._lock:
                self._fly_state = {}
            return base
        t = max(0.0, min(1.0, t))
        # ease-out cubic
        e = 1.0 - (1.0 - t) ** 3
        sx, sy = float(fly["sx"]), float(fly["sy"])
        ex, ey = float(fly["ex"]), float(fly["ey"])
        mx = (sx + ex) * 0.5
        my = min(sy, ey) - max(35.0, 0.14 * abs(ex - sx))
        u = 1.0 - e
        x = u * u * sx + 2 * u * e * mx + e * e * ex
        y = u * u * sy + 2 * u * e * my + e * e * ey
        out = base.copy()
        ring = fly.get("ring") or (80, 160, 255)
        cv2.line(
            out,
            (int(sx), int(sy)),
            (int(x), int(y)),
            ring,
            2,
            cv2.LINE_AA,
        )
        self._paste_chip_clipped(out, fly["chip"], int(x), int(y), ring)
        return out

    @staticmethod
    def _paste_chip_clipped(
        canvas: np.ndarray,
        chip: np.ndarray,
        cx: int,
        cy: int,
        ring_bgr: Tuple[int, int, int],
        border: int = 3,
    ) -> None:
        hh, ww = chip.shape[:2]
        x0, y0 = int(cx - ww // 2), int(cy - hh // 2)
        H, W = canvas.shape[:2]
        x1, y1 = x0 + ww, y0 + hh
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(W, x1), min(H, y1)
        if cx0 >= cx1 or cy0 >= cy1:
            return
        sx0, sy0 = cx0 - x0, cy0 - y0
        sx1, sy1 = sx0 + (cx1 - cx0), sy0 + (cy1 - cy0)
        region = canvas[cy0:cy1, cx0:cx1]
        chip_part = chip[sy0:sy1, sx0:sx1]
        np.copyto(region, chip_part)
        rx0, ry0 = max(0, x0), max(0, y0)
        rx1, ry1 = min(W - 1, x1 - 1), min(H - 1, y1 - 1)
        if rx0 < rx1 and ry0 < ry1:
            cv2.rectangle(canvas, (rx0, ry0), (rx1, ry1), ring_bgr, border, cv2.LINE_AA)

    def _resize_rgb_panel(self, rgb: np.ndarray, target_h: int) -> np.ndarray:
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        h, w = rgb.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((target_h, target_h, 3), dtype=np.uint8)
        scale = target_h / float(h)
        nw = max(1, int(w * scale))
        return cv2.resize(rgb, (nw, target_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _fmt_score(v: Optional[float]) -> str:
        if v is None:
            return "NA"
        return f"{float(v):.3f}"

    @classmethod
    def _status_lines_colored(
        cls, ctx: Optional[Dict[str, Any]]
    ) -> List[Tuple[Optional[str], Tuple[int, int, int]]]:
        """Key lines + BGR fill; None = blank spacer row between sections."""
        if not ctx:
            return [("?", (80, 80, 80))]
        mode = str(ctx.get("mode") or "")
        G = (55, 235, 55)
        # Vivid BGR for frame-vs-rep line (not dark gray)
        VS = (0, 200, 255)
        V = (100, 175, 255)
        S = (60, 140, 255)
        F = (100, 220, 220)
        U = (60, 200, 255)

        if mode == "new_cluster":
            cid = int(ctx.get("cluster_id", -1))
            fi = int(ctx.get("frame_idx", -1))
            sc = float(ctx.get("current_score", 0.0))
            mk = ctx.get("mismatch_kind")
            if mk == "visual":
                line2, c0 = "Visual mismatch (CLIP)", V
            elif mk == "spatial":
                line2, c0 = "Spatial mismatch (pose)", S
            elif mk == "first_frame":
                line2, c0 = "First frame (seed cluster)", F
            else:
                line2, c0 = "New cluster (other)", (200, 200, 70)
            return [
                (f"New memory cluster {cid}", c0),
                (None, c0),
                (line2, c0),
                (None, c0),
                (f"Frame {fi}  score {sc:.3f}", VS),
            ]

        if mode == "matched":
            cid = int(ctx.get("cluster_id", -1))
            fi = int(ctx.get("frame_idx", -1))
            cur = ctx.get("current_score")
            cmp_f = ctx.get("compared_rep_frame_idx")
            old_s = ctx.get("old_rep_score")
            upd = bool(ctx.get("representative_updated"))

            row2 = f"Frame {fi} score {cls._fmt_score(cur)}"
            if cmp_f is not None and old_s is not None:
                row2 += (
                    f"  vs representative frame {int(cmp_f)} "
                    f"score {cls._fmt_score(old_s)}"
                )

            out: List[Tuple[Optional[str], Tuple[int, int, int]]] = [
                (f"Matched to cluster {cid}", G),
                (None, G),
                (row2, VS),
                (None, VS),
            ]
            if upd:
                out.append(("Representative updated", U))
            else:
                out.append(("Representative unchanged", (165, 165, 165)))
            return out

        return [(mode or "?", (80, 80, 80))]

    @staticmethod
    def _wrap_text_to_width(
        text: str,
        font: int,
        scale: float,
        thick_fill: int,
        max_w: int,
    ) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        words = text.split()
        lines: List[str] = []
        cur = ""
        for w in words:
            trial = f"{cur} {w}".strip() if cur else w
            tw, _ = cv2.getTextSize(trial, font, scale, thick_fill)[0]
            if tw <= max_w:
                cur = trial
                continue
            if cur:
                lines.append(cur)
                cur = ""
            wtw, _ = cv2.getTextSize(w, font, scale, thick_fill)[0]
            if wtw <= max_w:
                cur = w
                continue
            chunk = ""
            for ch in w:
                t2 = chunk + ch
                tw2, _ = cv2.getTextSize(t2, font, scale, thick_fill)[0]
                if tw2 <= max_w:
                    chunk = t2
                else:
                    if chunk:
                        lines.append(chunk)
                    chunk = ch
            cur = chunk
        if cur:
            lines.append(cur)
        return lines

    def _status_lines_wrapped(
        self,
        viz_context: Optional[Dict[str, Any]],
        width: int,
        font: int,
        scale: float,
        thick_fill: int,
    ) -> List[Tuple[Optional[str], Tuple[int, int, int]]]:
        max_x = max(40, width - 24)
        out: List[Tuple[Optional[str], Tuple[int, int, int]]] = []
        for text, col in self._status_lines_colored(viz_context):
            if text is None:
                out.append((None, col))
                continue
            for seg in self._wrap_text_to_width(text, font, scale, thick_fill, max_x):
                out.append((seg, col))
        return out

    def _status_side_panel(
        self, min_height: int, width: int, viz_context: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.78
        thick_fill = 2
        thick_outline = 4
        lh = 32
        y0 = 28
        rows_wrapped = self._status_lines_wrapped(
            viz_context, width, font, scale, thick_fill
        )
        need_h = y0 + max(0, len(rows_wrapped)) * lh + 14
        height = max(min_height, need_h)
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 220
        y = y0
        max_y = height - 10
        for text, col in rows_wrapped:
            if y > max_y:
                break
            if text is None:
                y += lh
                continue
            cv2.putText(
                canvas,
                text,
                (12, y),
                font,
                scale,
                (0, 0, 0),
                thick_outline,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                text,
                (12, y),
                font,
                scale,
                col,
                thick_fill,
                cv2.LINE_AA,
            )
            y += lh
        return canvas

    def _make_cluster_thumbnail_tile(self, c: Any) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        th = self.thumb_h
        cid = int(c.get("id", -1))
        rep = c.get("representative")
        fidx = getattr(rep, "frame_idx", -1) if rep is not None else -1
        path = c.get("saved_path")
        im = None
        if path and os.path.isfile(path):
            im = cv2.imread(path)
        tile = np.ones((th, th, 3), dtype=np.uint8) * 230
        if im is not None and im.size > 0:
            ih, iw = im.shape[:2]
            sc = th / float(ih)
            tw = max(1, int(iw * sc))
            im = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
            if tw >= th:
                x0 = (tw - th) // 2
                im = im[:, x0 : x0 + th]
            else:
                pad = np.ones((th, th, 3), dtype=np.uint8) * 220
                x0 = (th - tw) // 2
                pad[:, x0 : x0 + tw] = im
                im = pad
            tile = im
        bcol = _cluster_color_bgr(cid)
        cv2.rectangle(tile, (0, 0), (th - 1, th - 1), bcol, 2)
        l1 = f"cluster {cid}"
        l2 = f"frame {fidx}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.38
        cv2.putText(tile, l1, (4, 14), font, fs, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(tile, l1, (4, 14), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, l2, (4, th - 8), font, fs, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(tile, l2, (4, th - 8), font, fs, (240, 240, 240), 1, cv2.LINE_AA)
        return tile, bcol

    def _cluster_cell_with_backing(self, c: Any) -> np.ndarray:
        """th×th tile plus three offset backing layers (two extra under the front card)."""
        tile, _bcol = self._make_cluster_thumbnail_tile(c)
        th = self.thumb_h
        mx = self._THUMB_BACKING_MAX_DX
        back_specs: List[Tuple[int, int, Tuple[int, int, int]]] = [
            (mx, mx, (145, 145, 150)),
            (6, 6, (152, 152, 157)),
            (3, 3, (158, 158, 163)),
        ]
        cw = th + mx
        ch = th + mx
        cell = np.ones((ch, cw, 3), dtype=np.uint8) * 235
        for dx, dy, fill_bgr in back_specs:
            cv2.rectangle(
                cell,
                (dx, dy),
                (dx + th - 1, dy + th - 1),
                fill_bgr,
                -1,
                cv2.LINE_AA,
            )
            cv2.rectangle(
                cell,
                (dx, dy),
                (dx + th - 1, dy + th - 1),
                (118, 118, 123),
                1,
                cv2.LINE_AA,
            )
        cell[0:th, 0:th] = tile
        return cell

    def _thumb_strip(
        self, manager: Any, strip_total_w: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        clusters = sorted(
            list(getattr(manager, "clusters", []) or []), key=lambda x: x.get("id", 0)
        )
        meta: Dict[str, Any] = {}
        strip_pad = 8
        strip_total_w = max(int(strip_total_w), 2 * strip_pad + 40)

        cell_w = self.thumb_h + self._THUMB_BACKING_MAX_DX
        cell_h = cell_w
        inner_w_target = strip_total_w - 2 * strip_pad
        inner_w_target = max(cell_w, inner_w_target)
        cols = max(1, inner_w_target // cell_w)
        if self.max_thumbs_per_row > 0:
            cols = min(cols, self.max_thumbs_per_row)

        if not clusters:
            h = self.thumb_h + 14
            row = np.ones((h, strip_total_w, 3), dtype=np.uint8) * 240
            cv2.putText(
                row,
                "No memory clusters yet",
                (20, row.shape[0] // 2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (120, 120, 120),
                2,
                cv2.LINE_AA,
            )
            return row, meta

        cells: List[np.ndarray] = []
        for c in clusters:
            cells.append(self._cluster_cell_with_backing(c))

        cell_h, cell_w = int(cells[0].shape[0]), int(cells[0].shape[1])
        order = [int(c.get("id", -1)) for c in clusters]

        grid_rows: List[np.ndarray] = []
        for i in range(0, len(cells), cols):
            chunk = cells[i : i + cols]
            row_img = np.hstack(chunk)
            if row_img.shape[1] < inner_w_target:
                row_img = cv2.copyMakeBorder(
                    row_img,
                    0,
                    0,
                    0,
                    inner_w_target - row_img.shape[1],
                    cv2.BORDER_CONSTANT,
                    value=(240, 240, 240),
                )
            grid_rows.append(row_img)
        inner = np.vstack(grid_rows)
        canvas = cv2.copyMakeBorder(
            inner,
            strip_pad,
            strip_pad,
            strip_pad,
            strip_pad,
            cv2.BORDER_CONSTANT,
            value=(235, 235, 235),
        )

        meta = {
            "pad": strip_pad,
            "cell_w": cell_w,
            "cell_h": cell_h,
            "cols": cols,
            "order": order,
        }
        return canvas, meta

    def _obs_beyond_depth_cutoff(self, obs: Any) -> bool:
        m = self._max_observation_depth_m
        if m <= 0:
            return False
        d = getattr(obs, "distance", None)
        try:
            df = float(d)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(df):
            return False
        return df > m

    @staticmethod
    def _is_no_class_token(s: str) -> bool:
        t = (s or "").strip().lower()
        if not t:
            return True
        return t in ("unknown", "unlabeled", "none", "n/a", "background")

    @staticmethod
    def _obs_has_semantic_class(obs: Any, class_lookup: List[str]) -> bool:
        """Draw only YOLO / detector boxes that resolve to a real class name."""
        name = getattr(obs, "class_name", None)
        if name is not None and str(name).strip():
            return not MemoryClusterLiveVisualizer._is_no_class_token(str(name))

        class_id = getattr(obs, "class_id", None)
        try:
            idx = int(class_id) if class_id is not None else -1
        except (TypeError, ValueError):
            return False
        if idx < 0 or not class_lookup or idx >= len(class_lookup):
            return False
        resolved = str(class_lookup[idx])
        return not MemoryClusterLiveVisualizer._is_no_class_token(resolved)

    @staticmethod
    def _class_name_for_obs(obs: Any, class_lookup: List[str]) -> str:
        name = getattr(obs, "class_name", None)
        if name:
            return str(name)
        class_id = getattr(obs, "class_id", None)
        try:
            idx = int(class_id) if class_id is not None else -1
        except (TypeError, ValueError):
            idx = -1
        if class_lookup and 0 <= idx < len(class_lookup):
            return str(class_lookup[idx])
        return f"class_{idx}"

    def _draw_detection_boxes(
        self,
        bgr: np.ndarray,
        observations: Optional[List[Any]],
        class_lookup: List[str],
        src_h: int,
        src_w: int,
    ) -> None:
        if not observations or not self._draw_yolo_boxes:
            return
        dh, dw = bgr.shape[:2]
        sh = max(int(src_h), 1)
        sw = max(int(src_w), 1)
        sx = dw / float(sw)
        sy = dh / float(sh)
        for obs in observations:
            if self._obs_beyond_depth_cutoff(obs):
                continue
            if not self._obs_has_semantic_class(obs, class_lookup):
                continue
            xyxy = getattr(obs, "xyxy", None)
            if xyxy is None:
                continue
            arr = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
            if arr.size < 4:
                continue
            x1f, y1f, x2f, y2f = float(arr[0, 0]), float(arr[0, 1]), float(arr[0, 2]), float(arr[0, 3])
            x1 = int(np.clip(round(x1f * sx), 0, dw - 1))
            y1 = int(np.clip(round(y1f * sy), 0, dh - 1))
            x2 = int(np.clip(round(x2f * sx), 0, dw - 1))
            y2 = int(np.clip(round(y2f * sy), 0, dh - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            try:
                cid = int(getattr(obs, "class_id", -1) or -1)
            except (TypeError, ValueError):
                cid = -1
            col = _cluster_color_bgr(max(0, cid))
            cv2.rectangle(bgr, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
            conf = float(getattr(obs, "conf", 0.0) or 0.0)
            raw_lbl = self._class_name_for_obs(obs, class_lookup)
            safe = "".join(c if 32 <= ord(c) < 127 else "_" for c in raw_lbl[:24])
            label = f"{safe} {conf:.2f}"
            ty = max(14, y1 - 2)
            cv2.putText(
                bgr,
                label,
                (x1, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                bgr,
                label,
                (x1, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                col,
                1,
                cv2.LINE_AA,
            )

    def _render(
        self,
        manager: Any,
        result: Optional[Dict[str, Any]],
        color_rgb: np.ndarray,
        frame_idx: int,
        observations: Optional[List[Any]] = None,
        viz_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        rgb = np.asarray(color_rgb)
        src_h, src_w = int(rgb.shape[0]), int(rgb.shape[1])

        cur = self._resize_rgb_panel(color_rgb, self.panel_h)
        cur_bgr = cv2.cvtColor(cur, cv2.COLOR_RGB2BGR)
        lookup = list(getattr(manager, "class_lookup", None) or [])
        self._draw_detection_boxes(
            cur_bgr, observations, lookup, src_h, src_w
        )

        side = self._status_side_panel(
            cur_bgr.shape[0], self.status_panel_w, viz_context
        )
        h1 = max(cur_bgr.shape[0], side.shape[0])
        if cur_bgr.shape[0] < h1:
            pad = h1 - cur_bgr.shape[0]
            cur_bgr = cv2.copyMakeBorder(
                cur_bgr, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)
            )
        if side.shape[0] < h1:
            pad = h1 - side.shape[0]
            side = cv2.copyMakeBorder(
                side, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(252, 252, 252)
            )

        top = np.hstack([cur_bgr, side])
        top_w = int(top.shape[1])
        thumbs_nat, thumb_stack = self._thumb_strip(manager, top_w)
        tw0 = int(thumbs_nat.shape[1])
        if tw0 > top_w:
            thumbs = cv2.resize(
                thumbs_nat,
                (top_w, int(thumbs_nat.shape[0])),
                interpolation=cv2.INTER_AREA,
            )
            thumb_stack["fly_scale_x"] = top_w / float(tw0)
        else:
            thumbs = thumbs_nat
            thumb_stack["fly_scale_x"] = 1.0
        if thumbs.shape[1] < top_w:
            pad = top_w - thumbs.shape[1]
            thumbs = cv2.copyMakeBorder(
                thumbs, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(240, 240, 240)
            )

        gap_px = 8
        gap = np.ones((gap_px, top_w, 3), dtype=np.uint8) * 200
        composite = np.vstack([top, gap, thumbs])
        layout: Dict[str, Any] = {
            "Wc": int(cur_bgr.shape[1]),
            "h1": int(h1),
            "gap": gap_px,
            "thumb_display_w": top_w,
            "has_thumb_strip": bool(
                len(list(getattr(manager, "clusters", []) or [])) > 0
            ),
            "thumb_stack": thumb_stack,
        }
        return composite, layout
