"""使用假摄像头测试生命周期，不访问任何真实采集设备。"""

from contextlib import ExitStack, redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import app
from posture import Config


class ArgumentTests(unittest.TestCase):
    def test_invalid_parameters_stop_before_camera_access(self):
        for arguments in (
            ["--camera", "https://example.invalid/camera"],
            ["--camera", "-1"], ["--threshold", "nan"],
            ["--threshold", "1"], ["--fps", "0"],
            ["--fps", "11"], ["--smoothing-seconds", "0.01"],
        ):
            with self.subTest(arguments=arguments), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    app.arguments(arguments)

    def test_missing_model_stops_before_loading_dependencies(self):
        with patch.object(app, "create_landmarker") as create, redirect_stderr(StringIO()):
            result = app.main(["--model", str(Path("__missing_model__.task"))])
        self.assertEqual(result, 1)
        create.assert_not_called()


class CameraLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cv2
            import numpy as np
            import mediapipe
        except ImportError as error:
            raise unittest.SkipTest("请使用项目 .venv 运行摄像头替身测试。") from error
        cls.cv2 = cv2
        cls.np = np

    def run_fake_camera(
        self, *, key=ord("q"), opened=True, missing=False, error=None,
        visible=1, black=False, grayscale=False, failed_with_frame=False,
    ):
        shape = (480, 640) if grayscale else (480, 640, 3)
        blank = self.np.full(shape, 0 if black else 100, dtype=self.np.uint8)
        capture = Mock()
        capture.isOpened.return_value = opened
        capture.grab.return_value = not missing
        capture.retrieve.return_value = (True, blank)
        if failed_with_frame:
            capture.retrieve.return_value = (False, blank)
        landmarker = Mock()
        landmarker.detect_for_video.return_value = SimpleNamespace(face_landmarks=[])
        landmarker.detect_for_video.side_effect = error
        args = SimpleNamespace(camera=0, backend="dshow")
        captured_readings = []

        def panel(frame, bounds, reading, *_):
            captured_readings.append(reading)
            return blank

        with ExitStack() as stack:
            stack.enter_context(patch.object(self.cv2, "VideoCapture", return_value=capture))
            for name in ("namedWindow", "resizeWindow", "imshow"):
                stack.enter_context(patch.object(self.cv2, name))
            destroyed = stack.enter_context(patch.object(self.cv2, "destroyAllWindows"))
            stack.enter_context(patch.object(self.cv2, "waitKey", return_value=key))
            stack.enter_context(patch.object(self.cv2, "getWindowProperty", return_value=visible))
            stack.enter_context(patch.object(app, "draw_panel", side_effect=panel))
            try:
                if error is not None:
                    with self.assertRaises(type(error)):
                        app.camera_loop(landmarker, args, Config(), None)
                elif not opened:
                    with self.assertRaises(app.UserError):
                        app.camera_loop(landmarker, args, Config(), None)
                else:
                    app.camera_loop(landmarker, args, Config(), None)
            finally:
                capture.release.assert_called_once()
                destroyed.assert_called_once()
        return captured_readings, landmarker

    def test_q_releases_camera(self):
        self.run_fake_camera()

    def test_escape_releases_camera(self):
        self.run_fake_camera(key=27)

    def test_window_close_releases_camera(self):
        self.run_fake_camera(key=255, visible=0)

    def test_failed_open_releases_camera(self):
        self.run_fake_camera(opened=False)

    def test_inference_error_releases_camera(self):
        self.run_fake_camera(error=RuntimeError("synthetic failure"))

    def test_keyboard_interrupt_releases_camera(self):
        self.run_fake_camera(error=KeyboardInterrupt())

    def test_camera_loss_is_unknown_and_never_inferred(self):
        readings, detector = self.run_fake_camera(missing=True)
        detector.detect_for_video.assert_not_called()
        self.assertEqual(readings[0].state, "无法判断")
        self.assertIn("摄像头画面丢失", readings[0].reason)

    def test_no_face_is_unknown(self):
        readings, _ = self.run_fake_camera()
        self.assertEqual(readings[0].state, "无法判断")

    def test_black_frame_is_reported_and_not_sent_to_inference(self):
        readings, detector = self.run_fake_camera(black=True)
        detector.detect_for_video.assert_not_called()
        self.assertEqual(readings[0].state, "无法判断")
        self.assertIn("画面过暗", readings[0].reason)

    def test_unexpected_frame_format_is_reported_without_inference(self):
        readings, detector = self.run_fake_camera(grayscale=True)
        detector.detect_for_video.assert_not_called()
        self.assertIn("画面格式异常", readings[0].reason)

    def test_failed_read_cannot_reuse_returned_pixels(self):
        readings, detector = self.run_fake_camera(failed_with_frame=True)
        detector.detect_for_video.assert_not_called()
        self.assertIn("摄像头画面丢失", readings[0].reason)

    def test_idle_frames_do_not_decode_or_infer_but_keyboard_still_works(self):
        blank = self.np.full((480, 640, 3), 100, dtype=self.np.uint8)
        capture = Mock()
        capture.isOpened.return_value = True
        capture.retrieve.return_value = (True, blank)
        clock = [0.0]

        def grab():
            clock[0] += 0.02
            return True

        capture.grab.side_effect = grab
        detector = Mock()
        detector.detect_for_video.return_value = SimpleNamespace(face_landmarks=[])
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.cv2, "VideoCapture", return_value=capture))
            stack.enter_context(patch.object(app.time, "monotonic", side_effect=lambda: clock[0]))
            stack.enter_context(patch.object(app, "draw_panel", return_value=blank))
            for name in ("namedWindow", "resizeWindow", "imshow", "destroyAllWindows"):
                stack.enter_context(patch.object(self.cv2, name))
            stack.enter_context(patch.object(self.cv2, "getWindowProperty", return_value=1))
            keyboard = stack.enter_context(
                patch.object(self.cv2, "waitKey", side_effect=[255] * 9 + [ord("q")])
            )
            app.camera_loop(detector, SimpleNamespace(camera=0, backend="dshow"), Config(), None)
        self.assertEqual(capture.grab.call_count, 10)
        self.assertEqual(keyboard.call_count, 10)
        self.assertEqual(capture.retrieve.call_count, 1)
        self.assertEqual(detector.detect_for_video.call_count, 1)
        capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
