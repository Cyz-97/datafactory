"""Smoke check for HistStaff's three one-dimensional drawing styles."""

import unittest
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from datafactory.hist import HistStaff


class HistPlotTest(unittest.TestCase):
    def test_styles_with_variable_bin_widths(self):
        centers = np.array([0.5, 2.0])
        values = np.array([2.0, 4.0])
        errors = np.array([0.25, 0.75])
        edges = np.array([0.0, 1.0, 3.0])
        hist = SimpleNamespace(
            name="sample", dimension=1,
            get_numpy=lambda: (centers, values, errors, edges),
        )

        for style in ("stairs", "errorbar", "hatched"):
            with self.subTest(style=style):
                fig, ax = plt.subplots()
                try:
                    if style == "stairs":
                        # The old third positional argument remains the stairs options.
                        result = HistStaff.plot(hist, "Energy", ax, {"color": "red"})
                    else:
                        result = HistStaff.plot(hist, "Energy", ax, style=style)
                    self.assertIs(result, ax)
                    self.assertEqual(ax.get_xlabel(), "Energy")
                    self.assertEqual(ax.get_legend_handles_labels()[1], ["$sample$"])

                    if style == "stairs":
                        self.assertEqual(ax.patches[0].get_edgecolor()[:3], (1.0, 0.0, 0.0))
                    elif style == "errorbar":
                        line = ax.containers[0].lines[0]
                        np.testing.assert_array_equal(line.get_xdata(), centers)
                        np.testing.assert_array_equal(line.get_ydata(), values)
                        self.assertEqual(line.get_marker(), "o")
                    elif style == "hatched":
                        bars = [patch for patch in ax.patches
                                if isinstance(patch, Rectangle)]
                        self.assertEqual(len(bars), 2)
                        np.testing.assert_allclose([bar.get_width() for bar in bars], [1, 2])
                        np.testing.assert_allclose([bar.get_y() for bar in bars], values-errors)
                        np.testing.assert_allclose([bar.get_height() for bar in bars], 2*errors)
                        self.assertTrue(all(bar.get_hatch() == "//////////" for bar in bars))
                finally:
                    plt.close(fig)

        with self.assertRaisesRegex(ValueError, "Unknown histogram plot style"):
            HistStaff.plot(hist, "Energy", style="invalid")
        hist.dimension = 2
        with self.assertRaisesRegex(ValueError, "only supports 1D"):
            HistStaff.plot(hist, "Energy", style="hatched")


if __name__ == "__main__":
    unittest.main()
