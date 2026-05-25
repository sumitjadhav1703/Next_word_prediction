from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DeploymentConfigTests(unittest.TestCase):
    def test_tensorflow_requirement_supports_streamlit_cloud_default_python(self):
        requirements = (PROJECT_ROOT / "requirements.txt").read_text().splitlines()
        tensorflow_lines = [line for line in requirements if line.startswith("tensorflow")]

        self.assertEqual(len(tensorflow_lines), 1)
        self.assertEqual(tensorflow_lines[0], "tensorflow==2.20.0")

    def test_runtime_txt_is_not_used_for_streamlit_cloud_python_selection(self):
        self.assertFalse((PROJECT_ROOT / "runtime.txt").exists())

    def test_numpy_requirement_supports_python_313_wheels(self):
        requirements = (PROJECT_ROOT / "requirements.txt").read_text().splitlines()
        numpy_lines = [line for line in requirements if line.startswith("numpy")]

        self.assertEqual(len(numpy_lines), 1)
        self.assertEqual(numpy_lines[0], "numpy>=2.1,<2.3")


if __name__ == "__main__":
    unittest.main()
