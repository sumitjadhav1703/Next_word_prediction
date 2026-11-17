"""Page behavior. Skips when Streamlit is not installed, so a bare unittest run still passes."""

import unittest
from pathlib import Path


try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover - deployment tests run without Streamlit
    AppTest = None


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _button(at, label):
    matches = [button for button in at.button if button.label == label]
    if len(matches) != 1:
        raise AssertionError(f"expected one {label!r} button, found {len(matches)}")
    return matches[0]


def _text(at):
    chunks = [block.value for block in (*at.markdown, *at.caption, *at.warning, *at.code)]
    return "\n".join(chunks)


@unittest.skipUnless(AppTest, "streamlit not installed")
class AppFlowTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP_PATH), default_timeout=60)
        self.app.run()

    def test_default_sentence_predicts_have(self):
        self.assertEqual(self.app.text_area[0].value, "the world as we")
        _button(self.app, "Predict").click().run()

        text = _text(self.app)
        self.assertIn("have", text)
        self.assertIn("1 of 3 requested next words appears after this phrase.", text)
        self.assertIn("the world as we have created it is a", text)
        self.assertFalse(self.app.exception)

    def test_example_replaces_input_and_predicts_ways(self):
        self.app.text_area[0].set_value("hello there").run()
        _button(self.app, "there are only two").click().run()

        self.assertEqual(self.app.text_area[0].value, "there are only two")
        _button(self.app, "Predict").click().run()
        self.assertIn("ways", _text(self.app))
        self.assertFalse(self.app.exception)

    def test_blank_input_warns(self):
        self.app.text_area[0].set_value("   ")
        _button(self.app, "Predict").click().run()

        warnings = [warning.value for warning in self.app.warning]
        self.assertIn("Enter some text before predicting the next word.", warnings)
        self.assertFalse(self.app.exception)

    def test_unknown_phrase_warns_without_crashing(self):
        self.app.text_area[0].set_value("qxqxqxqx notawordzz")
        _button(self.app, "Predict").click().run()

        warnings = [warning.value for warning in self.app.warning]
        self.assertTrue(any("No recorded next word" in warning for warning in warnings))
        self.assertFalse(self.app.exception)


if __name__ == "__main__":
    unittest.main()
