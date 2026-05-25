from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DeploymentConfigTests(unittest.TestCase):
    def test_streamlit_cloud_runtime_uses_tensorflow_supported_python(self):
        runtime = (PROJECT_ROOT / "runtime.txt").read_text().strip()

        self.assertEqual(runtime, "python-3.11")

    def test_tensorflow_is_required_for_deployment(self):
        requirements = (PROJECT_ROOT / "requirements.txt").read_text().splitlines()
        tensorflow_lines = [line for line in requirements if line.startswith("tensorflow")]

        self.assertEqual(len(tensorflow_lines), 1)
        self.assertNotIn("python_version", tensorflow_lines[0])


if __name__ == "__main__":
    unittest.main()
