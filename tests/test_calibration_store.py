"""校准数据持久化测试：只使用合成数值，不读取摄像头或真实人脸资料。"""

import json
from pathlib import Path
import tempfile
import unittest

from calibration_store import CalibrationStore
from mouth import MouthMonitor
from posture import Config, DistanceExperiment, Measurement


class CalibrationStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "calibration.json"
        self.context = dict(camera=0, backend="dshow", frame_width=640, frame_height=480)

    def tearDown(self):
        self.temp.cleanup()

    def test_saves_and_reloads_both_baselines(self):
        store = CalibrationStore(self.path, **self.context)
        store.save_posture(123.45)
        store.save_mouth((0.12, 0.13, 0.14, 0.25, 0.08))

        loaded = CalibrationStore(self.path, **self.context)
        self.assertAlmostEqual(loaded.posture_baseline, 123.45)
        self.assertEqual(loaded.mouth_baseline, (0.12, 0.13, 0.14, 0.25, 0.08))
        self.assertEqual(loaded.status, "已加载本地校准数据")

    def test_saving_one_baseline_preserves_the_other(self):
        store = CalibrationStore(self.path, **self.context)
        store.save_posture(123.45)
        store.save_mouth((0.12, 0.13, 0.14, 0.25, 0.08))
        store.save_posture(130.0)

        loaded = CalibrationStore(self.path, **self.context)
        self.assertEqual(loaded.posture_baseline, 130.0)
        self.assertEqual(loaded.mouth_baseline, (0.12, 0.13, 0.14, 0.25, 0.08))

    def test_camera_context_mismatch_does_not_load_baseline(self):
        store = CalibrationStore(self.path, **self.context)
        store.save_posture(123.45)
        other = CalibrationStore(self.path, camera=1, backend="dshow", frame_width=640, frame_height=480)
        self.assertIsNone(other.posture_baseline)
        self.assertIn("摄像头配置已变化", other.status)

    def test_corrupt_file_is_ignored(self):
        self.path.write_text("not-json", encoding="utf-8")
        loaded = CalibrationStore(self.path, **self.context)
        self.assertIsNone(loaded.posture_baseline)
        self.assertIsNone(loaded.mouth_baseline)
        self.assertIn("无法读取", loaded.status)

    def test_invalid_values_are_rejected(self):
        store = CalibrationStore(self.path, **self.context)
        with self.assertRaises(ValueError):
            store.save_posture(float("nan"))
        with self.assertRaises(ValueError):
            store.save_mouth((0.1, 0.2, 0.3, 0.4))

    def test_file_contains_only_calibration_numbers_and_context(self):
        store = CalibrationStore(self.path, **self.context)
        store.save_posture(123.45)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(payload),
            {
                "version", "camera", "backend", "frame_width", "frame_height",
                "updated_at", "posture_baseline_size", "mouth_baseline_features",
            },
        )
        self.assertFalse(self.path.with_name(self.path.name + ".tmp").exists())


class BaselineLoadTests(unittest.TestCase):
    def test_distance_experiment_accepts_valid_saved_baseline(self):
        experiment = DistanceExperiment(Config())
        self.assertTrue(experiment.load_baseline(120.0))
        self.assertEqual(experiment.baseline, 120.0)
        reading = experiment.update(1.0, Measurement(120, 144))
        self.assertEqual(reading.state, "无法判断")
        self.assertEqual(reading.reason, "正在积累稳定数据")

    def test_distance_experiment_rejects_invalid_saved_baseline(self):
        experiment = DistanceExperiment(Config())
        for value in (None, 0, -1, float("inf"), float("nan"), True):
            self.assertFalse(experiment.load_baseline(value))
        self.assertIsNone(experiment.baseline)

    def test_mouth_monitor_accepts_valid_saved_baseline(self):
        monitor = MouthMonitor()
        baseline = (0.12, 0.13, 0.14, 0.25, 0.08)
        self.assertTrue(monitor.load_baseline(baseline))
        self.assertEqual(monitor.baseline, baseline)

    def test_mouth_monitor_rejects_invalid_saved_baseline(self):
        monitor = MouthMonitor()
        for values in (None, (), (0.1, 0.2), (0.1, 0.2, 0.3, 0, 0.1), (0.1, 0.2, 0.3, 0.4, float("nan"))):
            self.assertFalse(monitor.load_baseline(values))
        self.assertIsNone(monitor.baseline)

    def test_interrupted_mouth_recalibration_keeps_previous_baseline(self):
        monitor = MouthMonitor()
        baseline = (0.12, 0.13, 0.14, 0.25, 0.08)
        self.assertTrue(monitor.load_baseline(baseline))
        monitor.calibrate(10.0)
        state = monitor.unknown("摄像头画面丢失")
        self.assertEqual(state, "无法判断")
        self.assertEqual(monitor.baseline, baseline)
        self.assertIn("继续使用上次校准基准", monitor.reason)


if __name__ == "__main__":
    unittest.main()
