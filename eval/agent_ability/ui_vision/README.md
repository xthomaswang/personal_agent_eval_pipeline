# ui_vision: can the model use a screen?

A probe of the visual skills a computer-use agent needs beyond reading a code: finding where to click,
filling a form from a profile, and reading a table. Synthetic pages with known ground truth, rendered
by WebKit on this Mac. Each model answers under three inputs:

- `vision`: the screenshot only;
- `vision+ocr`: the screenshot plus Apple Vision OCR lines with their boxes (Chinese pages are read with
  `zh-Hans,en-US`; Vision's default is English only);
- `ax`: no image, an accessibility-style list built from the DOM: the controls, headings, labels and
  table cells (role, name, placeholder, value, checked, box) and the other visible text, what a harness
  reading the accessibility tree would see.

Comparing them shows how much of a miss is seeing, and how much is understanding.

## Files

```
pages/<page>.html          a self-contained page (inline CSS, system fonts, no scripts, no network)
pages/<page>.tasks.json    its items (format below)
render.swift               page -> <out>.png + <out>.elements.json (boxes of data-qa elements, "ax" list)
ocr.swift                  png -> Apple Vision OCR lines with boxes (JSON lines)
validate.py                checks a page's tasks against its rendered elements
run.py                     asks the model, scores, writes runs/ui_vision/<run>/
```

Render and validate one page (from this directory):

```sh
swift render.swift pages/job_application.html /tmp/job 1
python3 validate.py pages/job_application.tasks.json /tmp/job.elements.json
```

## Pages

- Self-contained HTML with inline CSS and system font stacks (`-apple-system, "Helvetica Neue", Arial`;
  Chinese: `"PingFang SC"`). No scripts, no images from the network (inline SVG for icons), no real
  company names or logos; all people, companies and data are fictional (`example.com` addresses).
- The whole page fits a 1280x1024 viewport at 100% (no scrolling); `render.swift` warns otherwise.
- Look like the real thing: the layouts, spacing and controls of common job-application, spreadsheet,
  settings and checkout pages, including their hard cases (placeholder-only fields, two similar
  fields, icon-only buttons, a disabled field, helper text, a select showing "Select...", distractor
  buttons such as "Apply with LinkedIn", "Save draft", "Cancel").
- Every element an item refers to has a unique `data-qa` id. A control's state is in the markup
  (`checked`, `selected`, `value`, `aria-checked` on `role="switch"`), so the screenshot and the `ax`
  list agree.

## Items (`<page>.tasks.json`)

```json
{
  "page": "job_application",
  "language": "en",
  "items": [
    {"id": "click-01", "type": "click", "instruction": "Click the field for the applicant's email address.",
     "target": ["email"]},
    {"id": "fill-01", "type": "fill", "instruction": "Fill in this application for the applicant below. ...",
     "profile": {"legal name": "Lin Qiao", "email": "lin.qiao@example.com"},
     "expect": {"first-name": {"value": "Lin"}, "sponsor-no": {"checked": true}, "degree": {"option": "Master's"}},
     "forbid": ["preferred-name"]},
    {"id": "read-01", "type": "read", "question": "What is the total of the order?", "answer": "$128.40",
     "match": "number"}
  ]
}
```

- `click`: `target` lists the `data-qa` ids that count as a hit (a checkbox and its label, for
  example). The answer is a point; it hits when it falls inside one of the targets' boxes.
- `fill`: the answer is a list of actions (`type` text into a field, `click` a checkbox, radio or
  switch, `select` an option of a select). `expect` maps each control that must change to its
  outcome: `{"value": text}` typed, `{"checked": true}` clicked, or `{"option": text}` chosen.
  `forbid` lists controls that must stay untouched. A field is right when an action lands inside its
  box (or names its element in `ax`) with the right value; values compare case-insensitively with
  spaces collapsed, and phone numbers by their digits.
- `read`: `match` is `text` (case-insensitive, spaces collapsed, trailing punctuation dropped),
  `number` (the first number in the answer, within 0.005, ignoring `$`, `,` and `%`), or `set`
  (comma-separated items in any order).

Every item must have one correct answer that a careful person reads off the page without guessing.

## Scoring details

- Coordinates: both Qwen3.8 models answer in a 0-1000 frame whatever the prompt says, so runs use
  `--coords norm1000` (the prompt asks for 0-1000 and boxes in the OCR and `ax` lists are given in it).
- Replies are parsed leniently: code fences and surrounding text, Qwen grounding forms (`point_2d`,
  `bbox_2d`, `{"x": [x, y]}`), numbers written as strings, `{"actions": [...]}`, objects one per line,
  and two quote slips (`, y": 5` and `"y": 5"`). `requested_form` records whether a reply used exactly
  the requested form, so format errors stay visible.
- `--rescore <out>` re-scores a run's records with the current scorer, without new requests.

## Results, 2026-10-27

M5 Max (40-core GPU, 128 GB), macOS 27.2 (26B5101f), Xcode 27.1 (27A9275), oMLX 0.7.1rc1 (57620a3f)
with MTP on. Five pages, 91 items, `--scale 1 --coords norm1000 --samples 3`, sampling 1.0 / 0.95 /
20, thinking off:

```sh
python3 run.py --models Qwen3.8-27B-oQ5-mtp --coords norm1000 --samples 3 --out ../../../runs/ui_vision/<run>
```

| Model | Items | vision | vision+ocr | ax |
|---|---|---|---|---|
| Qwen3.8-27B oQ5 | click (56 x 3) | 165/168 | 166/168 | 168/168 |
| Qwen3.8-Flash-Next oQ4e | click | 163/168 | 157/168 (6 unparsed) | 167/168 (1 unparsed) |
| Qwen3.8-27B oQ5 | fill, items (6 x 3) | 13/18 | 13/18 | 16/18 |
| Qwen3.8-Flash-Next oQ4e | fill, items | 8/18 | 6/18 | 14/18 (1 unparsed) |
| Qwen3.8-27B oQ5 | fill, fields | 146/153 | 143/153 | 153/153 |
| Qwen3.8-Flash-Next oQ4e | fill, fields | 132/153 | 136/153 | 150/153 |
| Qwen3.8-27B oQ5 | read (29 x 3) | 81/87 | 82/87 | 84/87 |
| Qwen3.8-Flash-Next oQ4e | read | 83/87 | 81/87 | 85/87 |

- Coordinates matter most. Asked for pixels, the models clicked right 29/56 (27B) and 11/56
  (Flash-Next) times; asked for 0-1000, 54/56 each (56 click items, one sample).
- Clicking from the screenshot alone is close to the accessibility-tree ceiling for both models. Many
  of Flash-Next's vision+ocr misses are format, not localization: its six unparsed replies are prose or
  broken JSON, mostly cut at the 64-token limit. Of its five clicks on nothing, two are near misses on
  small targets (a sort arrow, a radio 2 px off), two are off target, and one is a pixel box in a 0-1000
  answer.
- Filling a form is the weak point, and the gap between the models is there: 27B 13/18 items fully right
  from the screenshot, Flash-Next 8/18. With the `ax` list both mostly succeed, so most misses come from
  the image being in context, not from the task:
  - Sponsorship radio: with the image, the models answer "Will you require visa sponsorship?" with the
    wrong option (27B on the applicant who needs it, 6/6; Flash-Next on both applicants, 11/12), although
    the clicks land squarely on an option. With `ax` both answer right.
  - Preferred first name: both models type it although it equals the legal name and the label says "If
    different from legal name" (27B 8/9 runs of that item, Flash-Next 4/9). This is harmless in
    practice; without this rule `ax` fill is 18/18 and 16/18.
  - Flash-Next only: switches, time fields and selects left wrong on the settings page, and a click on
    Continue (forbidden) on the checkout page.
- Reading is good except for arithmetic and small Chinese text. Summing the Travel rows of the expense
  table fails often: 27B 7/9 wrong across the three inputs; Flash-Next 6/6 wrong with the image, 3/3
  right from `ax`. 27B misreads the author name 刚落地的小王 (5/6 wrong with the image); Vision OCR drops
  的 there too, so OCR cannot fix it.
- OCR lines add little: vision+ocr is within a few items of vision for both models, and costs about 1900
  more prompt tokens per request.
- Both models emit malformed coordinate JSON at this sampling (27B `"y": "504"` and `"y": 557"`,
  Flash-Next `, y": 785`). MTP is not the cause: on four 27B click prompts x 25 samples, 23/100 replies
  slipped with MTP on and 30/100 with MTP off, and greedy decoding reproduces `"y": "504"`. A harness
  needs a lenient parser or constrained decoding for actions.

Run records stay local (`runs/ui_vision/final-27b-oq5`, `runs/ui_vision/final-flashnext-oq4e`). The
first runs, superseded, had an `ax` list without placeholders or static text (so `ax` could not see
format hints such as MM/YYYY) and OCR'd the Chinese page with English only. Both models were rerun with
the current `render.swift`, and the Chinese page's vision+ocr cells were rerun with `zh-Hans,en-US` OCR
and merged in. Latency is not reported: the three samples of an item share a prefix-cache hit.
