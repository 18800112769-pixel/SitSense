"""只用人工数字和合成轮廓测试，不读取相机或真实人脸资料。"""

import math
from types import SimpleNamespace
import unittest

from posture import Config, DistanceExperiment, FACE_OVAL, Measurement, measure_face


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.experiment = DistanceExperiment(self.config)

    def calibrate(self):
        self.experiment.start_calibration(0)
        for index in range(1, 41):
            reading = self.experiment.update(index / 8, Measurement(100, 144))
        self.assertEqual(reading.state, "无法判断")
        self.assertEqual(self.experiment.baseline, 120)

    def settle(self, ratio, start=5.125, count=10):
        for index in range(count):
            reading = self.experiment.update(start + index / 8, Measurement(100 * ratio, 144 * ratio))
        return reading

    def test_uncalibrated_is_never_normal(self):
        reading = self.experiment.update(0, Measurement(100, 144))
        self.assertEqual(reading.state, "无法判断")
        self.assertIsNone(reading.raw_ratio)

    def test_stable_calibration_and_approach_then_recovery(self):
        self.calibrate()
        normal = self.settle(1)
        self.assertEqual(normal.state, "正常")
        self.assertAlmostEqual(normal.smooth_ratio, 1)
        near = self.settle(1.2, 6.375)
        self.assertEqual(near.state, "疑似过近")
        self.assertAlmostEqual(near.smooth_ratio, 1.2)
        recovered = self.settle(1, 7.625)
        self.assertEqual(recovered.state, "正常")

    def test_threshold_boundary(self):
        self.calibrate()
        self.assertEqual(self.settle(1.15).state, "疑似过近")

    def test_minor_jitter_is_smoothed(self):
        self.calibrate()
        for index, ratio in enumerate((1, 1.01, 0.99, 1.02, 1)):
            reading = self.experiment.update(5.125 + index / 8, Measurement(100 * ratio, 144 * ratio))
        self.assertEqual(reading.state, "正常")
        self.assertAlmostEqual(reading.smooth_ratio, 1)

    def test_fast_change_is_unknown_before_settling(self):
        self.calibrate()
        self.settle(1)
        reading = self.experiment.update(6.375, Measurement(130, 187.2))
        self.assertEqual(reading.state, "无法判断")
        self.assertIsNone(reading.smooth_ratio)
        self.assertAlmostEqual(reading.raw_ratio, 1.3)

    def test_missing_face_clears_history_immediately(self):
        self.calibrate()
        self.settle(1.2)
        reading = self.experiment.update(6.375, None, "未检测到人脸")
        self.assertEqual(reading.state, "无法判断")
        self.assertIsNone(reading.smooth_ratio)
        reading = self.experiment.update(6.5, Measurement(100, 144))
        self.assertEqual(reading.state, "无法判断")
        self.assertIsNone(reading.smooth_ratio)

    def test_failed_recalibration_keeps_previous_baseline(self):
        self.calibrate()
        self.experiment.start_calibration(5.1)
        reading = self.experiment.update(5.125, None, "未检测到人脸")
        self.assertEqual(self.experiment.baseline, 120)
        self.assertIn("继续使用上次校准基准", reading.reason)

    def test_unstable_recalibration_keeps_previous_baseline(self):
        self.calibrate()
        self.experiment.start_calibration(5.1)
        for index in range(1, 41):
            reading = self.experiment.update(5.1 + index / 8, Measurement(100 + index, 144))
        self.assertEqual(self.experiment.baseline, 120)
        self.assertIn("继续使用上次校准基准", reading.reason)

    def test_successful_recalibration_replaces_previous_baseline(self):
        self.calibrate()
        self.experiment.start_calibration(5.1)
        for index in range(1, 41):
            reading = self.experiment.update(5.1 + index / 8, Measurement(110, 121))
        self.assertNotEqual(self.experiment.baseline, 120)
        self.assertIn("校准成功", reading.reason)

    def test_unstable_calibration_is_rejected(self):
        self.experiment.start_calibration(0)
        for index in range(1, 41):
            reading = self.experiment.update(index / 8, Measurement(100 + index, 144))
        self.assertIsNone(self.experiment.baseline)
        self.assertIn("变化过大", reading.reason)

    def test_width_height_changes_cannot_cancel_out(self):
        self.experiment.start_calibration(0)
        for index in range(1, 41):
            ratio = 1 + index / 100
            reading = self.experiment.update(index / 8, Measurement(100 * ratio, 144 / ratio))
        self.assertIn("校准失败", reading.reason)

    def test_too_few_samples_fail_calibration(self):
        self.experiment = DistanceExperiment(Config(detection_fps=8))
        self.experiment.start_calibration(0)
        for index in range(1, 21):
            reading = self.experiment.update(index / 4, Measurement(100, 144))
        self.assertIsNone(self.experiment.baseline)
        self.assertIn("采样不足", reading.reason)

    def test_five_fps_calibration_is_supported(self):
        self.experiment = DistanceExperiment(Config(detection_fps=5))
        self.experiment.start_calibration(0)
        for index in range(1, 26):
            self.experiment.update(index / 5, Measurement(100, 144))
        self.assertEqual(self.experiment.baseline, 120)

    def test_stale_gap_cannot_keep_an_old_state(self):
        self.calibrate()
        self.settle(1.2)
        reading = self.experiment.update(8, Measurement(120, 172.8))
        self.assertEqual(reading.state, "无法判断")
        self.assertIsNone(reading.smooth_ratio)

    def test_sampling_interruption_cancels_calibration(self):
        self.experiment.start_calibration(0)
        self.experiment.update(0.125, Measurement(100, 144))
        reading = self.experiment.update(2, Measurement(100, 144))
        self.assertIn("校准失败", reading.reason)
        self.assertIsNone(self.experiment.baseline)

    def test_slow_missing_samples_preserve_the_specific_failure(self):
        for reason in ("摄像头画面丢失", "画面过暗或接近全黑", "未检测到人脸"):
            with self.subTest(reason=reason):
                experiment = DistanceExperiment(self.config)
                experiment.update(0, None, reason)
                reading = experiment.update(1, None, reason)
                self.assertEqual(reading.state, "无法判断")
                self.assertEqual(reading.reason, reason)
                self.assertIsNone(reading.smooth_ratio)

    def test_interrupted_calibration_keeps_camera_failure_reason(self):
        self.experiment.start_calibration(0)
        self.experiment.update(0.125, Measurement(100, 144))
        reading = self.experiment.update(2, None, "画面过暗或接近全黑")
        self.assertIn("校准失败", reading.reason)
        self.assertIn("画面过暗", reading.reason)
        self.assertIsNone(self.experiment.baseline)

    def test_invalid_input_never_produces_a_ratio(self):
        self.calibrate()
        for index, value in enumerate((0, -1, math.nan, math.inf)):
            reading = self.experiment.update(5.125 + index / 8, Measurement(value, 144))
            self.assertEqual(reading.state, "无法判断")
            self.assertIsNone(reading.raw_ratio)

    def test_non_monotonic_time_is_unknown(self):
        self.calibrate()
        reading = self.experiment.update(4, Measurement(100, 144))
        self.assertEqual(reading.state, "无法判断")


class MeasurementTests(unittest.TestCase):
    def face(self, cx=0.5, cy=0.5, width=0.3, height=0.4):
        points = [SimpleNamespace(x=cx, y=cy) for _ in range(478)]
        for index, landmark in enumerate(FACE_OVAL):
            angle = index * 2 * math.pi / len(FACE_OVAL)
            points[landmark] = SimpleNamespace(
                x=cx + width / 2 * math.cos(angle),
                y=cy + height / 2 * math.sin(angle),
            )
        return points

    def measure(self, faces):
        return measure_face(faces, 640, 480, Config())

    def test_translation_does_not_change_size(self):
        first, _, _ = self.measure([self.face()])
        moved, _, _ = self.measure([self.face(cx=0.6, cy=0.6)])
        self.assertAlmostEqual(first.size, moved.size)

    def test_face_scaling_gives_expected_ratio(self):
        first, _, _ = self.measure([self.face()])
        larger, bounds, _ = self.measure([self.face(width=0.36, height=0.48)])
        self.assertAlmostEqual(larger.size / first.size, 1.2)
        self.assertEqual(len(bounds), 4)

    def test_missing_multiple_incomplete_small_and_clipped_faces(self):
        cases = (
            [], [self.face(), self.face()], [self.face()[:10]],
            [self.face(width=0.05)], [self.face(cx=0.05)],
        )
        for faces in cases:
            with self.subTest(face_count=len(faces)):
                measurement, bounds, reason = self.measure(faces)
                self.assertIsNone(measurement)
                self.assertIsNone(bounds)
                self.assertTrue(reason)

    def test_non_finite_landmark_is_rejected(self):
        points = self.face()
        points[FACE_OVAL[0]].x = math.nan
        self.assertIsNone(self.measure([points])[0])


if __name__ == "__main__":
    unittest.main()
