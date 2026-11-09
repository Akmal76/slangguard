import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from src.data.clean import LABELS, normalize_text

WEB = Path("web")
CORPUS = [
    "Enak, BANGET!!",
    "don't stop, it's fine",
    "ga' mau, jum'at 17's",
    "AMP amp &amp; camp ampas amp3 _amp",
    "teman-teman, benar-benar enak",
    "__laugh__ __sad__",
    "café naïve résumé",
    "😀 mantul 🔥🔥",
    "大家好 مرحبا",
    "１２３ ４５６",
    "é combining accent",
    "tab\tseparated\nand\r\nlines",
    "nbsp here and em space",
    "​zero width​",
    "İstanbul ß ǅ",
    "kuy!!! wkwkwk gaje bgt",
    "(test) [UNK] <b>tag</b> a--b x_y",
    "   ",
    "!!!",
    "",
    "ñamp ampñ éampé",
]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_the_page_cleans_text_exactly_like_python():
    script = (
        "import('./web/normalize.js').then(m => {"
        " const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
        " console.log(JSON.stringify(input.map(m.normalizeText))); })"
    )

    run = subprocess.run(
        ["node", "-e", script], input=json.dumps(CORPUS), capture_output=True, text=True, check=True
    )

    js_result = json.loads(run.stdout)
    differences = [
        (t, normalize_text(t), j)
        for t, j in zip(CORPUS, js_result, strict=True)
        if normalize_text(t) != j
    ]
    assert differences == []


def test_the_page_only_refers_to_files_that_exist():
    html = (WEB / "index.html").read_text()
    script = (WEB / "app.js").read_text()

    assert 'href="style.css"' in html and (WEB / "style.css").exists()
    assert 'src="app.js"' in html
    assert 'from "./normalize.js"' in script and (WEB / "normalize.js").exists()
    assert '"examples.json"' in script and (WEB / "examples.json").exists()


def test_transformers_js_is_pinned_to_a_version_that_works_in_browsers():
    """Version 4.3.2 failed in a real browser when loaded straight from the CDN: its default
    file broke the tokenizer, and its web build needs an import map. 3.8.1 was tested."""
    script = (WEB / "app.js").read_text()

    match = re.search(r"@huggingface/transformers@(\d+\.\d+\.\d+)\"", script)
    assert match, "the CDN import must name one exact version"
    assert match.group(1).startswith("3."), match.group(1)


def test_the_labels_in_the_page_are_the_labels_of_the_model():
    script = (WEB / "app.js").read_text()

    labels = json.loads(re.search(r"const LABELS = (\[.*?\]);", script).group(1))

    assert labels == list(LABELS)


def test_the_examples_are_a_short_list_of_sentences_with_words():
    examples = json.loads((WEB / "examples.json").read_text())

    assert 4 <= len(examples) <= 10
    assert all(isinstance(e, str) and normalize_text(e) for e in examples)


def test_the_space_readme_says_static():
    header = (WEB / "README.md").read_text().split("---")[1]

    assert "sdk: static" in header and "app_file: index.html" in header


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_exported_model_can_not_be_committed_by_accident():
    ignored = subprocess.run(["git", "check-ignore", "-q", "web/model/onnx/model_quantized.onnx"])

    assert ignored.returncode == 0
