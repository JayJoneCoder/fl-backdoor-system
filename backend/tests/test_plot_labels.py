"""图表标签回归测试：确保生成的图片使用中文标签和中文字体配置。"""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("plot_script", PROJECT_ROOT / "scripts" / "plot.py")
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


class PlotLabelsTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_chinese_font_and_metric_labels_are_configured(self):
        self.assertEqual(plt.rcParams["font.sans-serif"][0], "Microsoft YaHei")
        self.assertFalse(plt.rcParams["axes.unicode_minus"])
        self.assertEqual(plot._metric_label("detect_tp"), "真阳性 (TP)")
        self.assertEqual(plot._metric_label("agg_benign_removal_rate"), "良性客户端移除率")
        self.assertEqual(plot._round_label(3), "第 3 轮")

    def test_main_log_charts_use_chinese_axis_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "实验.csv"
            csv_path.write_text("round,accuracy,asr,loss\n0,0.5,0.1,0.9\n1,0.8,0.2,0.5\n",
                                encoding="utf-8")
            captured = []
            with patch.object(plot, "_savefig", side_effect=lambda path: captured.append((Path(path), plt.gcf()))):
                df = plot._safe_read_csv(csv_path)
                plot.plot_main_log(csv_path, df)

            self.assertEqual(len(captured), 4)
            accuracy_axes = captured[0][1].axes[0]
            self.assertEqual(accuracy_axes.get_xlabel(), "轮次")
            self.assertEqual(accuracy_axes.get_ylabel(), "准确率")
            self.assertIn("准确率曲线", accuracy_axes.get_title())
            self.assertEqual(captured[1][1].axes[0].get_ylabel(), "攻击成功率（ASR）")
            self.assertEqual(captured[2][1].axes[0].get_xlabel(), "攻击成功率（ASR）")
            self.assertEqual(captured[3][1].axes[0].get_ylabel(), "损失")


if __name__ == "__main__":
    unittest.main()
