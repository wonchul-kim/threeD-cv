#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag Pose Estimation Demo (Complete, from camera to overlay)
----------------------------------------------------------------

This script detects AprilTags in a live stream (Intel RealSense D4xx color stream,
a standard webcam, or a video file), estimates 6-DoF pose, and overlays:

- A 3D cube on the tag
- XYZ axes
- The tag's (x, y, z) translation in meters in the **camera frame**

It follows the approach described here:
- https://joonhyung-lee.github.io/blog/2023/apriltag-pose-estimation/

It supports two Python libraries:
1) pupil-apriltags  (preferred; easy to install wheels)
2) apriltag         (SWIG bindings; on some platforms you may need a compiler)

The code will try pupil-apriltags first; if not available, it will fall back to apriltag.

USAGE EXAMPLES
--------------
# RealSense (D435/D455...) color stream, tag size 10 cm
python apriltag_pose_demo.py --source realsense --tag-size 0.10

# Webcam at index 0 (⚠️ needs intrinsics; provide fx,fy,cx,cy or accept rough guess)
python apriltag_pose_demo.py --source 0 --tag-size 0.10 --fx 615 --fy 615 --cx 640 --cy 360

# Video file
python apriltag_pose_demo.py --source /path/to/video.mp4 --tag-size 0.10 --fx 615 --fy 615 --cx 640 --cy 360

NOTES
-----
- tag-size is the **outer black square side length** in meters. If you print a 10 cm tag, use 0.10.
- Coordinates shown are in the **camera coordinate system** (x right, y down, z forward).
- For accurate scale with non‑RealSense cameras, supply calibrated intrinsics (fx, fy, cx, cy).
- Press 'q' to quit, 's' to save a snapshot.
"""
import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import cv2


# -------------------------
# Library detection wrapper
# -------------------------
class ATDetector:
    """
    Wrapper that uses pupil_apriltags if available, otherwise falls back to apriltag.

    Unified output for each detection:
        - corners: (4,2) float32 array (pixel coords, CCW)
        - center:  (2,) float32 array (pixel coords)
        - R:       (3,3) rotation matrix (tag->camera)
        - t:       (3,1) translation vector in meters (tag center in camera frame)
        - tag_id:  int
    """

    def __init__(self, families: str = "tag36h11", nthreads: int = 2, quad_decimate: float = 1.0):
        self.impl = None
        self.detector = None

        # Try pupil_apriltags first
        try:
            from pupil_apriltags import Detector as PupilDetector  # type: ignore
            self.impl = "pupil"
            self.detector = PupilDetector(
                families=families,
                nthreads=nthreads,
                quad_decimate=quad_decimate,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0,
            )
        except Exception:
            # Fallback to apriltag
            try:
                import apriltag  # type: ignore

                self.impl = "apriltag"
                self.detector = apriltag.Detector(
                    apriltag.DetectorOptions(families=families, quad_decimate=quad_decimate)
                )
            except Exception as e:
                raise RuntimeError(
                    "Neither pupil_apriltags nor apriltag could be imported. Please install one:\n"
                    "  pip install pupil-apriltags\n"
                    "or\n"
                    "  pip install apriltag\n"
                ) from e

    def detect(
        self, gray: np.ndarray, K: np.ndarray, tag_size: float
    ) -> List[dict]:
        """Run detection and pose estimation.

        Args:
            gray: Grayscale image (uint8)
            K: 3x3 camera intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]
            tag_size: tag side length in meters

        Returns:
            List of dicts with keys: corners, center, R, t, tag_id
        """
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

        if self.impl == "pupil":
            # pupil_apriltags
            results = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(fx, fy, cx, cy),
                tag_size=tag_size,
            )
            out = []
            for r in results:
                R = np.asarray(r.pose_R, dtype=np.float32)
                t = np.asarray(r.pose_t, dtype=np.float32).reshape(3, 1)
                corners = np.asarray(r.corners, dtype=np.float32)
                center = np.asarray(r.center, dtype=np.float32)
                out.append(
                    {"corners": corners, "center": center, "R": R, "t": t, "tag_id": int(r.tag_id)}
                )
            return out

        elif self.impl == "apriltag":
            # apriltag (SWIG bindings)
            results = self.detector.detect(gray)
            out = []
            for r in results:
                pose_M, e0, e1 = self.detector.detection_pose(
                    r, (fx, fy, cx, cy), tag_size
                )
                pose_M = np.asarray(pose_M, dtype=np.float64)  # 4x4
                R = pose_M[:3, :3].astype(np.float32)
                t = pose_M[:3, 3].astype(np.float32).reshape(3, 1)
                corners = np.asarray(r.corners, dtype=np.float32)
                center = np.asarray(r.center, dtype=np.float32)
                tag_id = int(getattr(r, "tag_id", getattr(r, "id", -1)))
                out.append(
                    {"corners": corners, "center": center, "R": R, "t": t, "tag_id": tag_id}
                )
            return out

        else:
            raise RuntimeError("Detector not initialized")


# ----------------------
# Video / camera sources
# ----------------------
@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32)


class FrameSource:
    """Abstract frame source."""

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def intrinsics(self) -> Intrinsics:
        raise NotImplementedError

    def release(self) -> None:
        pass


class RealSenseSource(FrameSource):
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        try:
            import pyrealsense2 as rs  # type: ignore
        except Exception as e:
            raise RuntimeError("pyrealsense2 not installed") from e
        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16,   fps)
        self.profile = self.pipeline.start(self.config)
        # align depth to color
        self.align = rs.align(rs.stream.color)

        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self._intr = Intrinsics(fx=float(intr.fx), fy=float(intr.fy), cx=float(intr.ppx), cy=float(intr.ppy))

        # depth scale (meters per unit)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        c = aligned.get_color_frame()
        d = aligned.get_depth_frame()
        if not c or not d:
            return None, None
        color = np.asanyarray(c.get_data())
        depth = np.asanyarray(d.get_data())  # uint16 in native units
        # Convert native units to millimeters for saving as 16-bit PNG
        depth_mm = (depth.astype(np.float32) * self.depth_scale * 1000.0).round().astype(np.uint16)
        return color, depth_mm

    def intrinsics(self) -> Intrinsics:
        return self._intr

    def release(self) -> None:
        self.pipeline.stop()



class OpenCVSource(FrameSource):
    def __init__(
        self,
        src: Union[int, str] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        guess_fov_deg: float = 60.0,
    ):
        self.cap = cv2.VideoCapture(src)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Get frame size
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or (width or 640))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or (height or 480))

        # If intrinsics provided, use them; else create a rough guess
        if fx and fy and cx is not None and cy is not None:
            self._intr = Intrinsics(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))
        else:
            # Fallback heuristic (⚠️ not accurate; use proper calibration for real scale)
            f = (w / 2.0) / math.tan(math.radians(guess_fov_deg) / 2.0)
            self._intr = Intrinsics(fx=float(f), fy=float(f), cx=w / 2.0, cy=h / 2.0)
            print(
                "[WARN] Using a rough focal-length guess from FOV=%.1f°. "
                "Provide --fx --fy --cx --cy for accurate metric scale." % guess_fov_deg
            )

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        return frame if ok else None

    def intrinsics(self) -> Intrinsics:
        return self._intr

    def release(self) -> None:
        self.cap.release()


# --------------
# Draw utilities
# --------------
def draw_tag_outline(img: np.ndarray, corners: np.ndarray, color=(0, 255, 0), thickness: int = 2) -> None:
    pts = corners.astype(int).reshape(-1, 2)
    for i in range(4):
        p1 = tuple(pts[i])
        p2 = tuple(pts[(i + 1) % 4])
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def draw_axes(img: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, axis_len: float = 0.05) -> None:
    """Draw XYZ axes with length axis_len (meters) on the tag coordinate frame."""
    # Points in tag frame (origin and three axes)
    obj_pts = np.float32([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]])
    rvec, _ = cv2.Rodrigues(R)
    dcoeff = np.zeros(5)
    img_pts, _ = cv2.projectPoints(obj_pts, rvec, t, K, dcoeff)
    img_pts = img_pts.reshape(-1, 2).astype(int)

    O, Xp, Yp, Zp = img_pts
    cv2.line(img, tuple(O), tuple(Xp), (0, 0, 255), 2, cv2.LINE_AA)   # X - red
    cv2.line(img, tuple(O), tuple(Yp), (0, 255, 0), 2, cv2.LINE_AA)   # Y - green
    cv2.line(img, tuple(O), tuple(Zp), (255, 0, 0), 2, cv2.LINE_AA)   # Z - blue


def draw_cube(img: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, tag_size: float) -> None:
    """Draw a wireframe cube standing on top of the tag square."""
    s = tag_size / 2.0
    # 8 vertices: bottom square (z=0) and top square (z=-tag_size)
    obj_pts = np.float32(
        [
            [-s, -s, 0],
            [s, -s, 0],
            [s, s, 0],
            [-s, s, 0],
            [-s, -s, -tag_size],
            [s, -s, -tag_size],
            [s, s, -tag_size],
            [-s, s, -tag_size],
        ]
    )
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
        ]
    )

    rvec, _ = cv2.Rodrigues(R)
    dcoeff = np.zeros(5)
    img_pts, _ = cv2.projectPoints(obj_pts, rvec, t, K, dcoeff)
    img_pts = img_pts.reshape(-1, 2).astype(int)

    for i, j in edges:
        cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), (0, 255, 0), 2, cv2.LINE_AA)


def put_tag_text(
    img: np.ndarray,
    origin_xy: Tuple[int, int],
    text_lines: List[str],
    scale: float = 0.6,
    thickness: int = 2,
    color=(255, 255, 255),
    bg=True,
) -> None:
    """Draw multi-line text with an optional dark background for readability."""
    x, y = origin_xy
    font = cv2.FONT_HERSHEY_SIMPLEX
    if bg:
        # Compute bounding box
        sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in text_lines]
        box_w = max((w for (w, h) in sizes), default=0) + 10
        box_h = sum((h + 8 for (w, h) in sizes), 0) + 10
        cv2.rectangle(img, (x - 5, y - box_h + 5), (x - 5 + box_w, y + 5), (0, 0, 0), -1)
    # Draw lines
    y_cursor = y
    for t in text_lines:
        cv2.putText(img, t, (x, y_cursor), font, scale, color, thickness, cv2.LINE_AA)
        y_cursor += int(22 * scale) + 6

def draw_principal_point(img: np.ndarray, K: np.ndarray, size: int = 12) -> None:
    cx, cy = int(K[0, 2]), int(K[1, 2])
    H, W = img.shape[:2]
    cx = max(0, min(cx, W-1)); cy = max(0, min(cy, H-1))
    cv2.line(img, (cx - size, cy), (cx + size, cy), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy - size), (cx, cy + size), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 3, (0, 0, 0), -1, cv2.LINE_AA)


def format_xyz_m(t: np.ndarray) -> Tuple[float, float, float]:
    x, y, z = float(t[0, 0]), float(t[1, 0]), float(t[2, 0])
    return x, y, z

def format_idx(i: int) -> str:
    return f"{i:07d}"

def to_SE3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R.astype(np.float32)
    T[:3, 3] = t.reshape(3).astype(np.float32)
    return T

def save_matrix_txt(path: str, T: np.ndarray) -> None:
    # Save 4x4 or 3x3 as text, rows on newlines, space-separated
    np.savetxt(path, T, fmt="%.9f")

def ensure_dirs(root: str) -> Tuple[str,str,str]:
    rgb_dir   = os.path.join(root, "rgb")
    depth_dir = os.path.join(root, "depth")
    pose_dir  = os.path.join(root, "cam_in_ob")
    os.makedirs(rgb_dir,   exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pose_dir,  exist_ok=True)
    return rgb_dir, depth_dir, pose_dir

def main():
    ap = argparse.ArgumentParser(description="AprilTag Pose Estimation Demo")
    ap.add_argument("--source", type=str, default="realsense",
                    help="'realsense' for Intel RealSense color stream, integer index for webcam (e.g., '0'), or a video file path.")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--families", type=str, default="tag36h11")
    ap.add_argument("--tag-size", type=float, default=0.05, help="Tag side length in meters (e.g., 0.10 for 10 cm).")
    ap.add_argument("--fx", type=float, default=None, help="Camera fx (pixels).")
    ap.add_argument("--fy", type=float, default=None, help="Camera fy (pixels).")
    ap.add_argument("--cx", type=float, default=None, help="Camera cx (pixels).")
    ap.add_argument("--cy", type=float, default=None, help="Camera cy (pixels).")
    ap.add_argument("--guess-fov", type=float, default=60.0, help="If intrinsics are missing (non-RealSense), guess fx,fy from this horizontal FOV (deg).")
    ap.add_argument("--save-out", type=str, default="/wonchul/outputs/pose/pose.mp4", help="Optional output video path (.mp4/.avi).")
    ap.add_argument("--no-cube", action="store_true", help="Do not draw the cube overlay.")
    ap.add_argument("--no-axes", action="store_true", help="Do not draw XYZ axes.")
    ap.add_argument("--decimate", type=float, default=1.0, help="quad_decimate parameter for the detector (trade speed/accuracy).")
    ap.add_argument("--headless", default=False, action="store_true", help="Run without GUI windows (no cv2.imshow).")
    ap.add_argument("--print-pose", default=False, action="store_true", help="Print per-tag x,y,z to console.")
    ap.add_argument("--show-principal", default=True, action="store_true", help="Draw the camera principal point (cx, cy).")
    ap.add_argument("--show-cam-info", default=False, action="store_true", help="Draw the camera principal point (cx, cy).")
    ap.add_argument("--output-dir", default='/wonchul/outputs/pose/fp')
    args = ap.parse_args()

    # ------------------
    # Select frame source
    # ------------------
    if args.source.lower() == "realsense":
        src: FrameSource = RealSenseSource(width=args.width, height=args.height, fps=args.fps)
    else:
        # Try parsing as int index, otherwise as path
        try:
            cam_idx = int(args.source)
            source_id: Union[int, str] = cam_idx
        except ValueError:
            source_id = args.source

        src = OpenCVSource(
            source_id,
            width=args.width,
            height=args.height,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            guess_fov_deg=args.guess_fov,
        )

    intr = src.intrinsics()
    K = intr.K

    os.makedirs(args.output_dir, exist_ok=True)
    rgb_dir, depth_dir, pose_dir = ensure_dirs(args.output_dir)
    save_matrix_txt(os.path.join(args.output_dir, 'K.txt'))


    # ---------------
    # Init detector
    # ---------------
    detector = ATDetector(families=args.families, nthreads=2, quad_decimate=args.decimate)
    print(f"[INFO] Using detector backend: {detector.impl}")

    # ---------------
    # Video writer
    # ---------------
    writer = None
    gui_enabled = not args.headless
    if args.save_out:
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if args.save_out.lower().endswith(".mp4") else "XVID"))
        writer = cv2.VideoWriter(args.save_out, fourcc, float(args.fps), (args.width, args.height))

    # ---------------
    # Main loop
    # ---------------
    last = time.time()
    frame_count = 1
    fps_smoothed = 0.0

    while True:
        color, depth_mm = src.read()
        if color is None:
            print("[INFO] End of stream or frame not available.")
            break

        # Ensure size matches the assumed K
        h, w = color.shape[:2]
        # (Optional) If incoming size differs, you could scale K accordingly.

        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray, K, tag_size=args.tag_size)

        overlay = color.copy()
        for det_idx, det in enumerate(detections):
            corners = det["corners"]
            center = det["center"]
            R = det["R"]
            t = det["t"]
            tag_id = det["tag_id"]
            
            ob_in_cam = to_SE3(R, t)
            cam_in_ob = np.linalg.inv(ob_in_cam)
            
            stem = format_idx(frame_count)
            rgb_path  = os.path.join(rgb_dir,  f"{stem}.png")
            pose_path = os.path.join(pose_dir, f"{stem}.txt")
            cv2.imwrite(rgb_path, color)
            save_matrix_txt(pose_path, cam_in_ob)

            if depth_mm is not None:
                depth_path = os.path.join(depth_dir, f"{stem}.png")
                # depth_mm is uint16 millimeters (0 == invalid). Keep as 16-bit PNG.
                cv2.imwrite(depth_path, depth_mm)

            draw_tag_outline(overlay, corners, (0, 255, 255), 2)
            if not args.no_cube:
                draw_cube(overlay, K, R, t, tag_size=args.tag_size)
            if not args.no_axes:
                draw_axes(overlay, K, R, t, axis_len=args.tag_size * 0.5)

            x, y, z = format_xyz_m(t)
            # Put per-tag text near the tag
            txt = [f"ID {tag_id}  x:{x:+.3f} m  y:{y:+.3f} m  z:{z:+.3f} m"]
            # put_tag_text(overlay, (int(center[0]) + 10, int(center[1]) - 10), txt, scale=0.6, thickness=2)
            put_tag_text(overlay, (10, 10 + 22), txt, scale=0.4, thickness=1)
            if args.print_pose:
                print(f"ID {tag_id}: x={x:+.4f} m, y={y:+.4f} m, z={z:+.4f} m")

        # Global HUD (top-left)
        now = time.time()
        dt = now - last
        frame_count += 1
        if dt >= 0.5:
            fps_inst = frame_count / dt
            fps_smoothed = 0.9 * fps_smoothed + 0.1 * fps_inst if fps_smoothed > 0 else fps_inst
            last = now
            frame_count = 0

        if args.show_cam_info:
            hud_lines = [
                f"FX:{intr.fx:.1f}  FY:{intr.fy:.1f}  CX:{intr.cx:.1f}  CY:{intr.cy:.1f}",
                f"family:{args.families}  tag:{args.tag_size:.3f} m  det:{len(detections)}  FPS:{fps_smoothed:.1f}",
                "Coords in CAMERA frame (x right, y down, z forward)",
            ]
            put_tag_text(overlay, (10, 10 + 22 * len(hud_lines)), hud_lines, scale=0.6, thickness=2)
        if args.show_principal: draw_principal_point(overlay, K, size=12)

        if writer is not None:
            writer.write(overlay)

        if gui_enabled:
            try:
                cv2.imshow("AprilTag Pose Demo", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                if key == ord("s"):
                    ts = int(time.time())
                    snap_path = f"snap_{ts}.png"
                    cv2.imwrite(snap_path, overlay)
                    print(f"[INFO] Saved snapshot -> {snap_path}")
            except cv2.error as e:
                # GUI backend not available; switch to headless on the fly
                print(f"[WARN] HighGUI unavailable ({e}). Switching to --headless mode.")
                gui_enabled = False

    if writer is not None:
        writer.release()
    src.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
