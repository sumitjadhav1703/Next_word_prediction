from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DeploymentConfigTests(unittest.TestCase):
    def test_requirements_file_exists_and_uses_tensorflow_cpu(self):
        req_file = PROJECT_ROOT / "requirements.txt"
        self.assertTrue(req_file.exists())
        with open(req_file, "r") as f:
            content = f.read()
            self.assertIn("tensorflow-cpu", content)

    def test_runtime_txt_is_not_used_for_streamlit_cloud_python_selection(self):
        self.assertFalse((PROJECT_ROOT / "runtime.txt").exists())


if __name__ == "__main__":
    unittest.main()
