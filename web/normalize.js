// The same cleaning as normalize_text in src/data/clean.py. The model was trained on text that
// went through that function, so the page has to do exactly the same before it asks for a
// label. tests/test_web.py compares the two on many tricky strings.

// Python's \w is any Unicode letter or digit plus "_". JavaScript's \w is ASCII only, so the
// classes are written out.
const WORD = "\\p{L}\\p{N}_";

export function normalizeText(text) {
  let cleaned = text.toLowerCase().replaceAll("'", ""); // "don't" -> "dont"
  cleaned = cleaned.replace(new RegExp(`[^${WORD}\\s]`, "gu"), " "); // other marks become spaces
  cleaned = cleaned.replace(new RegExp(`(?<![${WORD}])amp(?![${WORD}])`, "gu"), " "); // left over from &amp;
  return cleaned.split(/\s+/).filter(Boolean).join(" ");
}
