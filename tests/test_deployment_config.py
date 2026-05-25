from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DeploymentConfigTests(unittest.TestCase):
    def test_requirements_file_is_omitted_for_streamlit_cloud_python_314(self):
        self.assertFalse((PROJECT_ROOT / "requirements.txt").exists())

    def test_tensorflow_is_not_required_on_streamlit_cloud_python_314(self):
        requirements = (PROJECT_ROOT / "requirements.txt")
        self.assertFalse(requirements.exists())

    def test_runtime_txt_is_not_used_for_streamlit_cloud_python_selection(self):
        self.assertFalse((PROJECT_ROOT / "runtime.txt").exists())

    def test_numpy_is_not_required_for_cloud_fallback_predictor(self):
        requirements = (PROJECT_ROOT / "requirements.txt")
        self.assertFalse(requirements.exists())


if __name__ == "__main__":
    unittest.main()
