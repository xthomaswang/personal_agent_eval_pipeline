# UI on real web pages, 2026-10-27 (protocol ui-real-1)

Can a local model click, fill in a form and read on real web pages? This run compares two models:

- Qwen3.8-27B oQ5 + MTP;
- Qwen3.8-Flash-Next oQ4e + MTP, a mixture-of-experts model.

The probe is [`eval/agent_ability/ui_vision/real.py`](../../../../../../../eval/agent_ability/ui_vision/real.py). It asks the model for an answer, then replays click and fill answers on the page in WebKit, so what is scored is what the page itself recorded. The same probe on five synthetic pages is in [the ui_vision README](../../../../../../../eval/agent_ability/ui_vision/README.md#results-2026-10-27).

`results.json` here holds every count. Below:

- [The set](#the-set)
- [Setup](#setup)
- [Results](#results)
- [What fails](#what-fails)
- [For a computer-use harness](#for-a-computer-use-harness)
- [Caveats](#caveats)

## The set

The set is private. Only the results are published.

- **Pages:** 75 pages captured from public websites. They cover job applications, sign-up and settings forms, tables and lists, articles and posts, toolbars with icon buttons, and code and small text.
  - 68 pages are in English and 7 in Chinese.
  - Each is a capture of about 1920x1000 with an offline replica page. The replica reproduces the capture's layout and records what is clicked and typed.
- **Items:** 449 in total.
  - **Click:** one target element.
  - **Fill:** a whole form, from a profile.
  - **Read:** a question answered from the page.
- **Audit:** every item was checked against both the capture and the replica.
  - 40 gold answers were corrected, mostly how an answer is matched (exact text, set or number) and fields that one instruction both asked to fill and to leave alone.
  - 19 items were excluded: 8 whose answer the scorer cannot match fairly, 5 ambiguous, 3 click targets the replica cannot record, 2 not fully visible and 1 with a wrong gold.
  - That leaves 430 items: 198 click, 24 fill and 208 read. The results use the audited gold. The set's original gold is lower in every cell except AX clicks, which are the same to one decimal; see the last table under Results.

## Setup

- **Machine:** M5 Max (40-core GPU, 128 GB), macOS 27.2 (26B5101f), Xcode 27.1 (27A9275).
- **Server:** oMLX 0.7.1rc1 (57620a3f).
  - MTP on, thinking off, 131072-token context.
  - Unconstrained decoding.
- **Sampling:** each model's agent sampling: temperature 1.0, top_p 0.95, top_k 20. Every item is asked 3 times.
- **Token limits:** click 64, fill 1500, read 128.
- **Coordinates:** answers use a 0–1000 frame (`--coords norm1000`).
- **Inputs (arms):**
  - **Screenshot:** the replica rendered in WebKit at the capture's size.
  - **Screenshot + OCR:** the same, plus Apple Vision OCR lines with their boxes. Pages with Chinese text are OCR'd as `zh-Hans,en-US`.
  - **AX list:** no image, an accessibility-style list of the replica's controls and text, each with its box.
  - **Original capture:** the original screenshot, scored on reads only. The replica matches the capture page-wide but not inline, so a click on the capture cannot be replayed on the replica.
- **Fill:** the profile is a fictional applicant: name, email, phone, address, LinkedIn, education, work authorization and visa sponsorship, how they heard of the job, and a birth date to use only if a form requires one.
  - The rule given with it: use only the profile, leave every other field untouched, and do not attach a file, save or submit.
  - The set's scorer requires every expected field's value, and requires protected fields to stay unchanged. Protected fields include credentials, consent and terms checkboxes, and fields the profile does not cover.
- **Read:**
  - **Strict** is the set's scorer: exact text after trimming (and after one pair of wrapping quotes is removed), a set of comma-separated items, or an exact number (a unit, if given, must match; `%` is required).
  - **Lenient** additionally ignores case, spacing, trailing punctuation and typography (dashes, quotes, primes, CJK brackets, a `#`, full-width forms).
  - Lenient also accepts an answer that ends with the gold after at most 12 extra leading characters, compares only the first number of a numeric answer, and accepts lists written with other separators.
- **Checks:** answering every item from the gold scores 430/430; deliberately wrong answers score 0.

## Results

Audited gold. Counts are item-samples: items x 3 samples.

| Model | Input | Click | Fill | Read, strict | Read, lenient |
|---|---|---|---|---|---|
| 27B oQ5 | screenshot | 582/594 (98.0%) | 46/72 (63.9%) | 563/624 (90.2%) | 595/624 (95.4%) |
| 27B oQ5 | screenshot + OCR | 571/594 (96.1%) | 52/72 (72.2%) | 565/624 (90.5%) | 598/624 (95.8%) |
| 27B oQ5 | AX list | 585/594 (98.5%) | 47/72 (65.3%) | 574/624 (92.0%) | 589/624 (94.4%) |
| 27B oQ5 | original capture | – | – | 559/624 (89.6%) | 584/624 (93.6%) |
| Flash-Next oQ4e | screenshot | 545/594 (91.8%) | 36/72 (50.0%) | 541/624 (86.7%) | 572/624 (91.7%) |
| Flash-Next oQ4e | screenshot + OCR | 549/594 (92.4%) | 41/72 (56.9%) | 518/624 (83.0%) | 555/624 (88.9%) |
| Flash-Next oQ4e | AX list | 586/594 (98.7%) | 45/72 (62.5%) | 550/624 (88.1%) | 571/624 (91.5%) |
| Flash-Next oQ4e | original capture | – | – | 534/624 (85.6%) | 568/624 (91.0%) |

- **Clicks from the AX list:** the two models are level.
- **From the screenshot:** 27B is clearly better at clicking (98.0% vs 91.8%; better on 34 items, worse on 4). It is ahead at filling (46 vs 36 of 72), but that rests on 24 items: better on 12, worse on 6, too few to be conclusive.
- **Filling a form** is the weakest skill for both models, whatever the input.
- **Reading** is mostly right. Most failures are format, not content (below).

The same runs against the set's original gold (449 items):

| Model | Input | Click | Fill | Read, strict | Read, lenient |
|---|---|---|---|---|---|
| 27B oQ5 | screenshot | 588/609 (96.6%) | 36/72 | 543/666 (81.5%) | 587/666 (88.1%) |
| 27B oQ5 | screenshot + OCR | 577/609 (94.7%) | 39/72 | 550/666 (82.6%) | 595/666 (89.3%) |
| 27B oQ5 | AX list | 600/609 (98.5%) | 34/72 | 564/666 (84.7%) | 584/666 (87.7%) |
| 27B oQ5 | original capture | – | – | 542/666 (81.4%) | 577/666 (86.6%) |
| Flash-Next oQ4e | screenshot | 550/609 (90.3%) | 24/72 | 518/666 (77.8%) | 559/666 (83.9%) |
| Flash-Next oQ4e | screenshot + OCR | 556/609 (91.3%) | 28/72 | 496/666 (74.5%) | 542/666 (81.4%) |
| Flash-Next oQ4e | AX list | 601/609 (98.7%) | 32/72 | 541/666 (81.2%) | 570/666 (85.6%) |
| Flash-Next oQ4e | original capture | – | – | 506/666 (76.0%) | 551/666 (82.7%) |

## What fails

Every click and fill failure in the three replica inputs was classified against the model's reply, the replay log and the rendered page. Every read failure in all four inputs was classified against the page's OCR lines and AX list.

### Clicks

| Failure | 27B: screenshot / + OCR / AX | Flash-Next: screenshot / + OCR / AX |
|---|---|---|
| near miss (within 25 px), on blank space or an adjacent control | 4 / 3 / 0 | 5 / 8 / 0 |
| the field's label instead of the input | 4 / 7 / 1 | 16 / 22 / 6 |
| far miss or a different control | 1 / 4 / 2 | 11 / 9 / 1 |
| reply not parsed, or cut off at 64 tokens | 1 / 7 / 6 | 13 / 4 / 0 |
| a point at (0, 0) | 0 / 0 / 0 | 2 / 0 / 0 |
| target box narrower than the visible item (set side) | 2 / 2 / 0 | 2 / 2 / 1 |
| **total** | 12 / 23 / 9 | 49 / 45 / 8 |

- **Flash-Next's screenshot misses are partly output discipline:** 15 of 49 are unparsed, cut off or at (0, 0).
  - Only 514 of its 594 replies were an `{"x", "y"}` object, and 239 of those added extra keys.
  - The other 80 were lists of labelled points, Qwen's `point_2d`/`bbox_2d`, or broken JSON. 14 hit the 64-token limit.
  - OCR lines fix the form (585/594) but not the aim: misses go from 49 to 45, and label clicks rise from 16 to 22.
- **Label clicks.** Items that ask for a field by its label are often answered by clicking the label above the input.
  - On the replica the label is not linked to its input, so this records nothing.
  - On a live site, a linked label would focus the input.
- **Near misses.** 56 of the 146 failures are within 25 px of the target, and 30 of those are label clicks.
  - Snapping to the nearest control would not be safe: 13 near misses already sit on a different control.
  - For example, both models hit an adjacent icon 20 px above the target in a vertical icon toolbar (27B 6 of 6).
- **Why 27B's screenshot + OCR is worse than its screenshot** (23 vs 12): 7 JSON syntax slips, such as `{"x": 575, 633}`, plus 3 more label clicks and 3 more far misses.
- **A more lenient parser would save 19 of the 31 format failures.** It would need to:
  - take the x and y of a reply cut off after them;
  - repair small JSON syntax slips such as `{"x": 575, 633}`;
  - read `points` before `bbox`;
  - accept a bare element index;
  - accept a tool-call wrapper.

### Fill

There are 279 failing field checks in all: 27B 128, Flash-Next 151.

| Failure (field checks / item-samples) | 27B | Flash-Next |
|---|---|---|
| a protected field changed | 23 / 18 | 39 / 34 |
| a field left empty, although the reply was parsed | 53 / 27 | 47 / 22 |
| wrong value format | 21 / 21 | 32 / 29 |
| reply not parsed | 28 / 6 | 23 / 8 |
| a value in the wrong field | 3 / 2 | 9 / 5 |
| select option not matched | 0 | 1 / 1 |

An item-sample can fail under several rows. In all, 71 of 27B's 216 and 94 of Flash-Next's 216 failed.

- **Protected fields are the main semantic failure of both models.**
  - Both invent a username and a password on a sign-up form.
  - 27B ticks a terms-of-service checkbox in 6 of 9 tries.
  - Flash-Next more often fills what the profile does not cover: it puts the school's name in "Current company" (6; 27B once) and ticks a pronoun (10).
  - Both answer an SMS-consent question (8).
- **Fields left empty:**
  - A Country select is skipped: the AX list shows it with no options.
  - 27B sometimes answers with a single action, as if in a step-by-step loop (8 items).
  - Flash-Next types at the label column instead of the inputs.
  - A third-party sign-in or apply button clicked first opens a dialog that takes the later typing (6 items).
- **Value format:**
  - "Preferred first name" is given as the full name (26 checks). The profile has no separate first name, so part of this is the profile's wording.
  - The location is given with the ZIP code or the whole street address where the form wants "City, ST" (18).
  - Item-samples that fail only on the full name or on a location that starts with "San Francisco, CA" are 4–7 per cell. Accepting them gives 27B 52 / 57 / 54 and Flash-Next 40 / 47 / 51 of 72 (screenshot / + OCR / AX), so formatting is not what separates the models.
- **Unparsed replies:**
  - 27B with OCR lines sometimes reasons in prose, with coordinates, before its JSON list.
  - Flash-Next writes coordinates such as `"x": [474, 465][0]`.
  - The runner also drops a `type` without coordinates after a click, and a `click` that carries text.
- **From the AX list,** Flash-Next is close to 27B (45 vs 47 of 72). Its extra screenshot failures are grounding and parsing.

### Reads

Of 588 strict read failures across the eight cells, 447 (76%) are format only: case, dashes and primes, extra or missing adjacent words, units and marks. Only 141 are real errors.

| | 27B screenshot | + OCR | AX | capture | Flash-Next screenshot | + OCR | AX | capture |
|---|---|---|---|---|---|---|---|---|
| strict failures | 61 | 59 | 50 | 65 | 83 | 106 | 74 | 90 |
| real errors | 11 | 8 | 16 | 14 | 18 | 30 | 25 | 19 |
| right apart from format | 98.2% | 98.7% | 97.4% | 97.8% | 97.1% | 95.2% | 96.0% | 97.0% |

- **The lenient check recovers 228 of the 447 format-only failures** and never accepts a real error. The rest are longer extra or missing words around a right answer, or marks such as `*`, `+` and `%`.
- **Real misreads come in three kinds:**
  - **Small Chinese text and emoji.** A 13-px line of Chinese with an emoji fails all 9 screenshot answers of both models and passes 3 of 3 from the AX list.
  - **A neighbouring element.** For example, a tab's counter instead of the asked-for count, or a heading instead of the text under it.
  - **Dropped suffixes.**
- **OCR lines** are neutral for 27B (real errors 11 to 8) and hurt Flash-Next (18 to 30). Flash-Next copies OCR mistakes, such as `fleld`, picks neighbouring lines, and once answered with a box copied from the OCR list (twice more from the AX list).
- **The AX list removes character errors entirely** in both models. Its fragments cause 10–11 real errors per model: headings split in two, inline link text missing from its sentence, code split into tokens, a required-field `*` as its own node.
- **The original capture is barely harder than the replica** (real errors +3 for 27B, +1 for Flash-Next). No arithmetic errors appeared.

## For a computer-use harness

- **Target by the accessibility tree when there is one.** Both models then click at 98.5% or better.
  - On a name collision, prefer interactive roles (text box, link, button) over text and headings. That would fix 7 of Flash-Next's 8 remaining AX misses.
- **Parse actions leniently and in a fixed order, or constrain decoding.**
  - Format costs 31 click item-samples, 14 unparsed fills, and 15 fills where actions were dropped or the reply held a single action.
  - Accept a single action, a click that carries text, and a `type` with no point (send it to the focused field).
  - Give clicks more than 64 tokens.
- **Close the loop:**
  - observe again after any click that changes the page;
  - check that a point resolves to a control, and move a label hit to its input;
  - keep asking until the form is done.
- **Guard the semantics in the harness, not the prompt.** Ask before touching credentials, consent, terms, pronoun and marketing controls, third-party sign-in, or any field the profile does not cover. Both models do these despite the prompt's rule.
- **Keep profile data per field:** first, last and preferred first name, and the city and state for a location.
- **Reads:**
  - judge by content, not exact string;
  - zoom into small CJK text;
  - do not give Flash-Next raw OCR lines.

## Caveats

- Fill has 24 items, so one item-sample is 1.4 points.
- The capture input scores reads only (see Setup).
- Fill uses a second version of the profile. The first gave the name as "Mira Chen (legal and preferred name)", and some replies typed the note into name fields. With the first profile, fill was:
  - 27B: 49 / 47 / 42 of 72;
  - Flash-Next: 40 / 44 / 47 of 72 (screenshot / + OCR / AX).
- The Flash-Next run was resumed once, after oMLX's prefill memory guard began refusing requests about 40 minutes into the run.
  - The model is 73.8 GB, and the server's footprint had grown from 84 to over 86 GB.
  - Records are deduplicated, so nothing is counted twice or lost.
- Latency is not reported: the three samples of an item share a prefix-cache hit.
