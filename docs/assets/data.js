window.BENCHMARK_DATA = {
  "repoUrl": "https://github.com/xthomaswang/personal_agent_eval_pipeline",
  "latest": "2026-10-30",
  "buildCount": 6,
  "resultCount": 20,
  "machineCount": 1,
  "harnessCount": 3,
  "machines": [
    {
      "id": "m5max-40c-128gb",
      "label": "M5 Max \u00b7 40-core GPU \u00b7 128 GB"
    }
  ],
  "protocols": [
    {
      "machine": "m5max-40c-128gb",
      "suite": "agent_ability",
      "id": "aider12-v0",
      "kind": "archive",
      "title": "Aider, 12 Python exercises x 3 harnesses, one run each (thinking off)",
      "latest": "2026-10-22",
      "metrics": [
        "Total",
        "Claude Code",
        "Hermes",
        "localmelo"
      ]
    },
    {
      "machine": "m5max-40c-128gb",
      "suite": "agent_ability",
      "id": "capability-1",
      "kind": "ability",
      "title": "Claude Code, Hermes, localmelo on Aider 12 + MCPMark easy 10 (gate set) and Aider 22 (ranking set); thinking off",
      "latest": "2026-10-26",
      "metrics": [
        "Claude Code (gate)",
        "Hermes (gate)",
        "localmelo (gate)",
        "Gate set",
        "Ranking set",
        "M7 tool-call cases"
      ]
    },
    {
      "machine": "m5max-40c-128gb",
      "suite": "agent_ability",
      "id": "capability-2",
      "kind": "ability",
      "title": "Claude Code, Hermes, localmelo on Aider 12 + MCPMark easy 10 (gate set) and Aider 22 (ranking set); thinking off",
      "latest": "2026-10-28",
      "metrics": [
        "Claude Code (gate)",
        "Hermes (gate)",
        "localmelo (gate)",
        "Gate set",
        "Ranking set",
        "M7 tool-call cases"
      ]
    },
    {
      "machine": "m5max-40c-128gb",
      "suite": "agent_ability",
      "id": "capability-3",
      "kind": "ability",
      "title": "Claude Code, Hermes, localmelo on Aider 12 + MCPMark easy 10 (gate set) and Aider 22 (ranking set); thinking off",
      "latest": "2026-10-30",
      "metrics": [
        "Claude Code (gate)",
        "Hermes (gate)",
        "localmelo (gate)",
        "Gate set",
        "Ranking set",
        "M7 tool-call cases"
      ]
    },
    {
      "machine": "m5max-40c-128gb",
      "suite": "agent_ability",
      "id": "ui-real-1",
      "kind": "ui",
      "title": "UI on real web pages: 75 pages, 430 items x 3 samples; click, fill a form, read; from the screenshot or the accessibility list",
      "latest": "2026-10-27",
      "metrics": [
        "Click, screenshot",
        "Click, AX list",
        "Fill, screenshot",
        "Fill, AX list",
        "Read, screenshot",
        "Read, AX list",
        "Read, original capture"
      ]
    },
    {
      "machine": "m5max-40c-128gb",
      "suite": "backend",
      "id": "quant-fidelity-v0",
      "kind": "archive",
      "title": "Quantization fidelity against BF16, and screenshot reading",
      "latest": "2026-10-22",
      "metrics": [
        "KL, all stable positions",
        "Top-1 agreement",
        "KL, agent assistant turns",
        "Top-1, agent turns",
        "Screenshot codes read"
      ]
    },
    {
      "machine": "m5max-40c-128gb",
      "suite": "backend",
      "id": "speed-1",
      "kind": "speed",
      "title": "Speed on oMLX 0.7.1rc1 (M5 Max): text-only, agent sampling, thinking off",
      "latest": "2026-10-26",
      "metrics": [
        "64K cold TTFT",
        "Cached-prefix TTFT",
        "Decode 16K",
        "Decode 32K",
        "Decode 64K",
        "Decode 110K",
        "20-step agent trajectory",
        "Peak server memory"
      ]
    }
  ],
  "rows": [
    {
      "schema": 1,
      "suite": "agent_ability",
      "protocol": "capability-1",
      "title": "Claude Code, Hermes, localmelo on Aider 12 + MCPMark easy 10 (gate set) and Aider 22 (ranking set); thinking off",
      "model": "Nanbeige4.3-33B oQ5",
      "family": "nanbeige4.3_33b",
      "quant": "oq5",
      "run_id": "20261026-104645",
      "date": "2026-10-26",
      "system": {
        "machine": "m5max-40c-128gb",
        "macos_build": "26B5101f",
        "xcode_build": "27A9275",
        "server": "oMLX 0.7.1rc1 (7a5a3d75)"
      },
      "metrics": [
        {
          "name": "Claude Code (gate)",
          "value": "16/22"
        },
        {
          "name": "Hermes (gate)",
          "value": "15/18 (partial)"
        },
        {
          "name": "localmelo (gate)",
          "value": "\u2014"
        },
        {
          "name": "Gate set",
          "value": "31/40 (partial)"
        },
        {
          "name": "Ranking set",
          "value": "\u2014"
        },
        {
          "name": "M7 tool-call cases",
          "value": "17/22"
        }
      ],
      "notes": "Capability leg stopped (interrupted) after 40 of 66 gate-set runs. Not run: localmelo, the ranking set.",
      "details": "../../backend/20261026-104645/README.md",
      "_dir": "docs/benchmark/m5max-40c-128gb/nanbeige4.3_33b/oq5/agent_ability/20261026-104645",
      "_details": "docs/benchmark/m5max-40c-128gb/nanbeige4.3_33b/oq5/backend/20261026-104645/README.md",
      "_harness_source": "docs/benchmark/m5max-40c-128gb/nanbeige4.3_33b/oq5/backend/20261026-104645/capability/results.json",
      "_harnesses": {
        "claude_code": {
          "gate": {
            "passed": 16,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 151.8,
            "modelCallsMedian": 12.5
          },
          "ranking": {
            "passed": 0,
            "runs": 0,
            "expected": 22,
            "modelTimeMedian": null,
            "modelCallsMedian": null
          }
        },
        "hermes": {
          "gate": {
            "passed": 15,
            "runs": 18,
            "expected": 22,
            "modelTimeMedian": 132.4,
            "modelCallsMedian": 7.5
          },
          "ranking": {
            "passed": 0,
            "runs": 0,
            "expected": 22,
            "modelTimeMedian": null,
            "modelCallsMedian": null
          }
        },
        "localmelo": {
          "gate": {
            "passed": 0,
            "runs": 0,
            "expected": 22,
            "modelTimeMedian": null,
            "modelCallsMedian": null
          },
          "ranking": {
            "passed": 0,
            "runs": 0,
            "expected": 22,
            "modelTimeMedian": null,
            "modelCallsMedian": null
          }
        }
      }
    },
    {
      "schema": 1,
      "suite": "backend",
      "protocol": "speed-1",
      "title": "Speed on oMLX 0.7.1rc1 (M5 Max): text-only, agent sampling, thinking off",
      "model": "Nanbeige4.3-33B oQ5",
      "family": "nanbeige4.3_33b",
      "quant": "oq5",
      "run_id": "20261026-104645",
      "date": "2026-10-26",
      "system": {
        "machine": "m5max-40c-128gb",
        "macos_build": "26B5101f",
        "xcode_build": "27A9275",
        "server": "oMLX 0.7.1rc1 (7a5a3d75)"
      },
      "metrics": [
        {
          "name": "64K cold TTFT",
          "value": "210.6 s"
        },
        {
          "name": "Cached-prefix TTFT",
          "value": "5.05 s"
        },
        {
          "name": "Decode 16K",
          "value": "15.9 tok/s"
        },
        {
          "name": "Decode 32K",
          "value": "13.5 tok/s"
        },
        {
          "name": "Decode 64K",
          "value": "10.5 tok/s"
        },
        {
          "name": "Decode 110K",
          "value": "9.0 tok/s"
        },
        {
          "name": "20-step agent trajectory",
          "value": "328.5 s"
        },
        {
          "name": "Peak server memory",
          "value": "74.0 GiB"
        }
      ],
      "notes": "Kept with a minor disturbance flag (eval/backend/DESIGN.md \u00a75.4): Cached-prefix TTFT. Some trajectory steps opened with a tool call, so their TTFT includes its decode.",
      "details": "README.md",
      "_dir": "docs/benchmark/m5max-40c-128gb/nanbeige4.3_33b/oq5/backend/20261026-104645",
      "_details": "docs/benchmark/m5max-40c-128gb/nanbeige4.3_33b/oq5/backend/20261026-104645/README.md"
    },
    {
      "schema": 1,
      "model": "Qwen3.8-27B oQ4 + MTP",
      "family": "qwen3.8_27b",
      "quant": "oq4",
      "run_id": "2026-10-21-quant-compare",
      "date": "2026-10-21",
      "system": {
        "machine": "m5max-40c-128gb",
        "macos_build": "26B5101f",
        "xcode_build": "27A9275",
        "server": "oMLX 0.7.1rc1 + G2b patch (57620a3f), MTP on"
      },
      "suite": "agent_ability",
      "protocol": "aider12-v0",
      "title": "Aider, 12 Python exercises x 3 harnesses, one run each (thinking off)",
      "metrics": [
        {
          "name": "Total",
          "value": "27/36"
        },
        {
          "name": "Claude Code",
          "value": "7/12"
        },
        {
          "name": "Hermes",
          "value": "11/12"
        },
        {
          "name": "localmelo",
          "value": "9/12"
        }
      ],
      "notes": "From the 2026-10-21/22 quantization comparison: the capability leg's 12 Aider gate exercises, once per harness, run with the harness-smoke suite (now eval/agent_ability) at its defaults of 600 s, 30 steps and 40 turns per task (capability-1 allows 3600 s), on a patched oMLX build (G2b, 57620a3f). The patched server makes this table a record rather than a protocol you can rerun from here.",
      "_dir": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq4/agent_ability/2026-10-21-quant-compare"
    },
    {
      "schema": 1,
      "model": "Qwen3.8-27B oQ4 + MTP",
      "family": "qwen3.8_27b",
      "quant": "oq4",
      "run_id": "2026-10-21-quant-compare",
      "date": "2026-10-21",
      "system": {
        "machine": "m5max-40c-128gb",
        "macos_build": "26B5101f",
        "xcode_build": "27A9275",
        "server": "oMLX 0.7.1rc1 + G2b patch (57620a3f), MTP on"
      },
      "suite": "backend",
      "protocol": "quant-fidelity-v0",
      "title": "Quantization fidelity against BF16, and screenshot reading",
      "metrics": [
        {
          "name": "KL, all stable positions",
          "value": "0.216"
        },
        {
          "name": "Top-1 agreement",
          "value": "91.7%"
        },
        {
          "name": "KL, agent assistant turns",
          "value": "0.016"
        },
        {
          "name": "Top-1, agent turns",
          "value": "96.2%"
        },
        {
          "name": "Screenshot codes read",
          "value": "54/60"
        }
      ],
      "notes": "From the 2026-10-21/22 quantization comparison: KL divergence and top-1 agreement of each quantization's next-token distribution against BF16, and screenshot codes read out of 60. It ran with earlier scripts, not in this repository, on a patched oMLX build, so this table is a record rather than a protocol you can rerun from here.",
      "_dir": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq4/backend/2026-10-21-quant-compare"
    },
    {
      "schema": 1,
      "model": "Qwen3.8-27B oQ5 + MTP",
      "family": "qwen3.8_27b",
      "quant": "oq5",
      "run_id": "2026-10-21-quant-compare",
      "date": "2026-10-22",
      "system": {
        "machine": "m5max-40c-128gb",
        "macos_build": "26B5101f",
        "xcode_build": "27A9275",
        "server": "oMLX 0.7.1rc1 + G2b patch (57620a3f), MTP on"
      },
      "suite": "agent_ability",
      "protocol": "aider12-v0",
      "title": "Aider, 12 Python exercises x 3 harnesses, one run each (thinking off)",
      "metrics": [
        {
          "name": "Total",
          "value": "32/36"
        },
        {
          "name": "Claude Code",
          "value": "9/12"
        },
        {
          "name": "Hermes",
          "value": "12/12"
        },
        {
          "name": "localmelo",
          "value": "11/12"
        }
      ],
      "notes": "From the 2026-10-21/22 quantization comparison: the capability leg's 12 Aider gate exercises, once per harness, run with the harness-smoke suite (now eval/agent_ability) at its defaults of 600 s, 30 steps and 40 turns per task (capability-1 allows 3600 s), on a patched oMLX build (G2b, 57620a3f). The patched server makes this table a record rather than a protocol you can rerun from here.",
      "_dir": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq5/agent_ability/2026-10-21-quant-compare"
    },
    {
      "schema": 1,
      "suite": "agent_ability",
      "protocol": "capability-1",
      "title": "Claude Code, Hermes, localmelo on Aider 12 + MCPMark easy 10 (gate set) and Aider 22 (ranking set); thinking off",
      "model": "Qwen3.8-27B oQ5 + MTP",
      "family": "qwen3.8_27b",
      "quant": "oq5",
      "run_id": "20261026-062952",
      "date": "2026-10-26",
      "system": {
        "machine": "m5max-40c-128gb",
        "macos_build": "26B5101f",
        "xcode_build": "27A9275",
        "server": "oMLX 0.7.1rc1 (7a5a3d75)"
      },
      "metrics": [
        {
          "name": "Claude Code (gate)",
          "value": "21/22"
        },
        {
          "name": "Hermes (gate)",
          "value": "20/22"
        },
        {
          "name": "localmelo (gate)",
          "value": "22/22"
        },
        {
          "name": "Gate set",
          "value": "63/66"
        },
        {
          "name": "Ranking set",
          "value": "61/66"
        },
        {
          "name": "M7 tool-call cases",
          "value": "21/22"
        }
      ],
      "reference": true,
      "details": "../../backend/20261026-062952/capability/results.json",
      "_dir": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq5/agent_ability/20261026-062952",
      "_details": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq5/backend/20261026-062952/capability/results.json",
      "_harness_source": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq5/backend/20261026-062952/capability/results.json",
      "_harnesses": {
        "claude_code": {
          "gate": {
            "passed": 21,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 24.9,
            "modelCallsMedian": 6.0
          },
          "ranking": {
            "passed": 18,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 73.9,
            "modelCallsMedian": 10.0
          }
        },
        "hermes": {
          "gate": {
            "passed": 20,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 20.6,
            "modelCallsMedian": 5.0
          },
          "ranking": {
            "passed": 21,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 64.5,
            "modelCallsMedian": 9.0
          }
        },
        "localmelo": {
          "gate": {
            "passed": 22,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 18.9,
            "modelCallsMedian": 4.5
          },
          "ranking": {
            "passed": 22,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 30.1,
            "modelCallsMedian": 6.0
          }
        }
      }
    },
    {
      "schema": 1,
      "suite": "agent_ability",
      "protocol": "ui-real-1",
      "title": "UI on real web pages: 75 pages, 430 items x 3 samples; click, fill a form, read; from the screenshot or the accessibility list",
      "model": "Qwen3.8-27B oQ5 + MTP",
      "family": "qwen3.8_27b",
      "quant": "oq5",
      "run_id": "20261027-ui-real",
      "date": "2026-10-27",
      "reference": true,
      "system": {
        "machine": "m5max-40c-128gb",
        "macos_build": "26B5101f",
        "xcode_build": "27A9275",
        "server": "oMLX 0.7.1rc1 (57620a3f)"
      },
      "metrics": [
        {
          "name": "Click, screenshot",
          "value": "582/594"
        },
        {
          "name": "Click, AX list",
          "value": "585/594"
        },
        {
          "name": "Fill, screenshot",
          "value": "46/72"
        },
        {
          "name": "Fill, AX list",
          "value": "47/72"
        },
        {
          "name": "Read, screenshot",
          "value": "563/624"
        },
        {
          "name": "Read, AX list",
          "value": "574/624"
        },
        {
          "name": "Read, original capture",
          "value": "559/624"
        }
      ],
      "notes": "Private set of replica pages, audited gold; reads by exact match. Screenshot + OCR, lenient reads and the failure analysis are in details.",
      "details": "README.md",
      "_dir": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq5/agent_ability/20261027-ui-real",
      "_details": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq5/agent_ability/20261027-ui-real/README.md",
      "_ui": {
        "replica:vision": {
          "click": [
            582,
            594
          ],
          "fill": [
            46,
            72
          ],
          "read_strict": [
            563,
            624
          ],
          "read_lenient": [
            595,
            624
          ]
        },
        "replica:vision+ocr": {
          "click": [
            571,
            594
          ],
          "fill": [
            52,
            72
          ],
          "read_strict": [
            565,
            624
          ],
          "read_lenient": [
            598,
            624
          ]
        },
        "replica:ax": {
          "click": [
            585,
            594
          ],
          "fill": [
            47,
            72
          ],
          "read_strict": [
            574,
            624
          ],
          "read_lenient": [
            589,
            624
          ]
        },
        "capture:vision": {
          "read_strict": [
            559,
            624
          ],
          "read_lenient": [
            584,
            624
          ]
        }
      },
      "_ui_source": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq5/agent_ability/20261027-ui-real/results.json",
      "_ui_meta": {
        "pages": 75,
        "items": {
          "click": 198,
          "fill": 24,
          "read": 208
        },
        "totalItems": 430,
        "samples": 3
      }
    },
    {
      "schema": 1,
      "suite": "agent_ability",
      "protocol": "capability-2",
      "title": "Claude Code, Hermes, localmelo on Aider 12 + MCPMark easy 10 (gate set) and Aider 22 (ranking set); thinking off",
      "model": "Qwen3.8-27B oQ5 + MTP",
      "family": "qwen3.8_27b",
      "quant": "oq5",
      "run_id": "20261028-211816",
      "date": "2026-10-28",
      "system": {
        "machine": "m5max-40c-128gb",
        "macos_build": "26B123",
        "xcode_build": "27A9275",
        "server": "oMLX 0.7.1rc1 (7a5a3d75)"
      },
      "metrics": [
        {
          "name": "Claude Code (gate)",
          "value": "20/22"
        },
        {
          "name": "Hermes (gate)",
          "value": "21/22"
        },
        {
          "name": "localmelo (gate)",
          "value": "22/22"
        },
        {
          "name": "Gate set",
          "value": "63/66"
        },
        {
          "name": "Ranking set",
          "value": "58/66"
        },
        {
          "name": "M7 tool-call cases",
          "value": "\u2014"
        }
      ],
      "details": "../../backend/20261028-211816/capability/results.json",
      "_dir": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq5/agent_ability/20261028-211816",
      "_details": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq5/backend/20261028-211816/capability/results.json",
      "_harness_source": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq5/backend/20261028-211816/capability/results.json",
      "_harnesses": {
        "claude_code": {
          "gate": {
            "passed": 20,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 23.0,
            "modelCallsMedian": 5.5
          },
          "ranking": {
            "passed": 19,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 50.8,
            "modelCallsMedian": 6.0
          }
        },
        "hermes": {
          "gate": {
            "passed": 21,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 23.2,
            "modelCallsMedian": 5.0
          },
          "ranking": {
            "passed": 18,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 45.0,
            "modelCallsMedian": 6.5
          }
        },
        "localmelo": {
          "gate": {
            "passed": 22,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 21.5,
            "modelCallsMedian": 4.0
          },
          "ranking": {
            "passed": 21,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 38.5,
            "modelCallsMedian": 6.0
          }
        }
      }
    },
    {
      "schema": 1,
      "suite": "agent_ability",
      "protocol": "capability-3",
      "title": "Claude Code, Hermes, localmelo on Aider 12 + MCPMark easy 10 (gate set) and Aider 22 (ranking set); thinking off",
      "model": "Qwen3.8-27B oQ5 + MTP",
      "family": "qwen3.8_27b",
      "quant": "oq5",
      "run_id": "20261030-230008",
      "date": "2026-10-30",
      "system": {
        "machine": "m5max-40c-128gb",
        "macos_build": "26B123",
        "xcode_build": "27B52",
        "server": "oMLX 0.7.1rc1 (7a5a3d75)"
      },
      "metrics": [
        {
          "name": "Claude Code (gate)",
          "value": "21/22"
        },
        {
          "name": "Hermes (gate)",
          "value": "21/22"
        },
        {
          "name": "localmelo (gate)",
          "value": "21/22"
        },
        {
          "name": "Gate set",
          "value": "63/66"
        },
        {
          "name": "Ranking set",
          "value": "57/66"
        },
        {
          "name": "M7 tool-call cases",
          "value": "\u2014"
        }
      ],
      "details": "../../backend/20261030-230008/capability/results.json",
      "_dir": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq5/agent_ability/20261030-230008",
      "_details": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq5/backend/20261030-230008/capability/results.json",
      "_harness_source": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq5/backend/20261030-230008/capability/results.json",
      "_harnesses": {
        "claude_code": {
          "gate": {
            "passed": 21,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 21.9,
            "modelCallsMedian": 6.0
          },
          "ranking": {
            "passed": 16,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 68.8,
            "modelCallsMedian": 10.0
          }
        },
        "hermes": {
          "gate": {
            "passed": 21,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 19.9,
            "modelCallsMedian": 5.0
          },
          "ranking": {
            "passed": 21,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 47.6,
            "modelCallsMedian": 7.0
          }
        },
        "localmelo": {
          "gate": {
            "passed": 21,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 25.3,
            "modelCallsMedian": 5.0
          },
          "ranking": {
            "passed": 20,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 53.1,
            "modelCallsMedian": 7.0
          }
        }
      }
    },
    {
      "schema": 1,
      "model": "Qwen3.8-27B oQ5 + MTP",
      "family": "qwen3.8_27b",
      "quant": "oq5",
      "run_id": "2026-10-21-quant-compare",
      "date": "2026-10-22",
      "system": {
        "machine": "m5max-40c-128gb",
        "macos_build": "26B5101f",
        "xcode_build": "27A9275",
        "server": "oMLX 0.7.1rc1 + G2b patch (57620a3f), MTP on"
      },
      "suite": "backend",
      "protocol": "quant-fidelity-v0",
      "title": "Quantization fidelity against BF16, and screenshot reading",
      "metrics": [
        {
          "name": "KL, all stable positions",
          "value": "0.093"
        },
        {
          "name": "Top-1 agreement",
          "value": "95.4%"
        },
        {
          "name": "KL, agent assistant turns",
          "value": "0.005"
        },
        {
          "name": "Top-1, agent turns",
          "value": "97.9%"
        },
        {
          "name": "Screenshot codes read",
          "value": "54/60"
        }
      ],
      "notes": "From the 2026-10-21/22 quantization comparison: KL divergence and top-1 agreement of each quantization's next-token distribution against BF16, and screenshot codes read out of 60. It ran with earlier scripts, not in this repository, on a patched oMLX build, so this table is a record rather than a protocol you can rerun from here.",
      "_dir": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq5/backend/2026-10-21-quant-compare"
    },
    {
      "schema": 1,
      "suite": "backend",
      "protocol": "speed-1",
      "title": "Speed on oMLX 0.7.1rc1 (M5 Max): text-only, agent sampling, thinking off",
      "model": "Qwen3.8-27B oQ5 + MTP",
      "family": "qwen3.8_27b",
      "quant": "oq5",
      "run_id": "20261026-062952",
      "date": "2026-10-26",
      "system": {
        "machine": "m5max-40c-128gb",
        "macos_build": "26B5101f",
        "xcode_build": "27A9275",
        "server": "oMLX 0.7.1rc1 (7a5a3d75)"
      },
      "metrics": [
        {
          "name": "64K cold TTFT",
          "value": "111.4 s"
        },
        {
          "name": "Cached-prefix TTFT",
          "value": "3.20 s"
        },
        {
          "name": "Decode 16K",
          "value": "55.1 tok/s"
        },
        {
          "name": "Decode 32K",
          "value": "51.0 tok/s"
        },
        {
          "name": "Decode 64K",
          "value": "41.9 tok/s"
        },
        {
          "name": "Decode 110K",
          "value": "36.9 tok/s"
        },
        {
          "name": "20-step agent trajectory",
          "value": "133.4 s"
        },
        {
          "name": "Peak server memory",
          "value": "58.0 GiB"
        }
      ],
      "reference": true,
      "details": "speed/units.json",
      "_dir": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq5/backend/20261026-062952",
      "_details": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq5/backend/20261026-062952/speed/units.json"
    },
    {
      "schema": 1,
      "model": "Qwen3.8-27B oQ6 + MTP",
      "family": "qwen3.8_27b",
      "quant": "oq6",
      "run_id": "2026-10-21-quant-compare",
      "date": "2026-10-22",
      "system": {
        "machine": "m5max-40c-128gb",
        "macos_build": "26B5101f",
        "xcode_build": "27A9275",
        "server": "oMLX 0.7.1rc1 + G2b patch (57620a3f), MTP on"
      },
      "suite": "agent_ability",
      "protocol": "aider12-v0",
      "title": "Aider, 12 Python exercises x 3 harnesses, one run each (thinking off)",
      "metrics": [
        {
          "name": "Total",
          "value": "31/36"
        },
        {
          "name": "Claude Code",
          "value": "9/12"
        },
        {
          "name": "Hermes",
          "value": "11/12"
        },
        {
          "name": "localmelo",
          "value": "11/12"
        }
      ],
      "notes": "From the 2026-10-21/22 quantization comparison: the capability leg's 12 Aider gate exercises, once per harness, run with the harness-smoke suite (now eval/agent_ability) at its defaults of 600 s, 30 steps and 40 turns per task (capability-1 allows 3600 s), on a patched oMLX build (G2b, 57620a3f). The patched server makes this table a record rather than a protocol you can rerun from here.",
      "_dir": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq6/agent_ability/2026-10-21-quant-compare"
    },
    {
      "schema": 1,
      "model": "Qwen3.8-27B oQ6 + MTP",
      "family": "qwen3.8_27b",
      "quant": "oq6",
      "run_id": "2026-10-21-quant-compare",
      "date": "2026-10-22",
      "system": {
        "machine": "m5max-40c-128gb",
        "macos_build": "26B5101f",
        "xcode_build": "27A9275",
        "server": "oMLX 0.7.1rc1 + G2b patch (57620a3f), MTP on"
      },
      "suite": "backend",
      "protocol": "quant-fidelity-v0",
      "title": "Quantization fidelity against BF16, and screenshot reading",
      "metrics": [
        {
          "name": "KL, all stable positions",
          "value": "0.043"
        },
        {
          "name": "Top-1 agreement",
          "value": "97.3%"
        },
        {
          "name": "KL, agent assistant turns",
          "value": "0.002"
        },
        {
          "name": "Top-1, agent turns",
          "value": "98.4%"
        },
        {
          "name": "Screenshot codes read",
          "value": "56/60"
        }
      ],
      "notes": "From the 2026-10-21/22 quantization comparison: KL divergence and top-1 agreement of each quantization's next-token distribution against BF16, and screenshot codes read out of 60. It ran with earlier scripts, not in this repository, on a patched oMLX build, so this table is a record rather than a protocol you can rerun from here.",
      "_dir": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq6/backend/2026-10-21-quant-compare"
    },
    {
      "schema": 1,
      "model": "Qwen3.8-27B oQ8 + MTP",
      "family": "qwen3.8_27b",
      "quant": "oq8",
      "run_id": "2026-10-21-quant-compare",
      "date": "2026-10-21",
      "system": {
        "machine": "m5max-40c-128gb",
        "macos_build": "26B5101f",
        "xcode_build": "27A9275",
        "server": "oMLX 0.7.1rc1 + G2b patch (57620a3f), MTP on"
      },
      "suite": "agent_ability",
      "protocol": "aider12-v0",
      "title": "Aider, 12 Python exercises x 3 harnesses, one run each (thinking off)",
      "metrics": [
        {
          "name": "Total",
          "value": "33/36"
        },
        {
          "name": "Claude Code",
          "value": "10/12"
        },
        {
          "name": "Hermes",
          "value": "11/12"
        },
        {
          "name": "localmelo",
          "value": "12/12"
        }
      ],
      "notes": "From the 2026-10-21/22 quantization comparison: the capability leg's 12 Aider gate exercises, once per harness, run with the harness-smoke suite (now eval/agent_ability) at its defaults of 600 s, 30 steps and 40 turns per task (capability-1 allows 3600 s), on a patched oMLX build (G2b, 57620a3f). The patched server makes this table a record rather than a protocol you can rerun from here.",
      "_dir": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq8/agent_ability/2026-10-21-quant-compare"
    },
    {
      "schema": 1,
      "model": "Qwen3.8-27B oQ8 + MTP",
      "family": "qwen3.8_27b",
      "quant": "oq8",
      "run_id": "2026-10-21-quant-compare",
      "date": "2026-10-21",
      "system": {
        "machine": "m5max-40c-128gb",
        "macos_build": "26B5101f",
        "xcode_build": "27A9275",
        "server": "oMLX 0.7.1rc1 + G2b patch (57620a3f), MTP on"
      },
      "suite": "backend",
      "protocol": "quant-fidelity-v0",
      "title": "Quantization fidelity against BF16, and screenshot reading",
      "metrics": [
        {
          "name": "KL, all stable positions",
          "value": "0.013"
        },
        {
          "name": "Top-1 agreement",
          "value": "98.5%"
        },
        {
          "name": "KL, agent assistant turns",
          "value": "0.001"
        },
        {
          "name": "Top-1, agent turns",
          "value": "99.2%"
        },
        {
          "name": "Screenshot codes read",
          "value": "54/60"
        }
      ],
      "notes": "From the 2026-10-21/22 quantization comparison: KL divergence and top-1 agreement of each quantization's next-token distribution against BF16, and screenshot codes read out of 60. It ran with earlier scripts, not in this repository, on a patched oMLX build, so this table is a record rather than a protocol you can rerun from here.",
      "_dir": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq8/backend/2026-10-21-quant-compare"
    },
    {
      "schema": 1,
      "suite": "agent_ability",
      "protocol": "capability-1",
      "title": "Claude Code, Hermes, localmelo on Aider 12 + MCPMark easy 10 (gate set) and Aider 22 (ranking set); thinking off",
      "model": "Qwen3.8-Flash-Next oQ4e + MTP",
      "family": "qwen3.8_flash_next",
      "quant": "oq4e",
      "run_id": "20261026-201922",
      "date": "2026-10-26",
      "system": {
        "machine": "m5max-40c-128gb",
        "macos_build": "26B5101f",
        "xcode_build": "27A9275",
        "server": "oMLX 0.7.1rc1 (7a5a3d75)"
      },
      "metrics": [
        {
          "name": "Claude Code (gate)",
          "value": "20/22"
        },
        {
          "name": "Hermes (gate)",
          "value": "22/22"
        },
        {
          "name": "localmelo (gate)",
          "value": "22/22"
        },
        {
          "name": "Gate set",
          "value": "64/66"
        },
        {
          "name": "Ranking set",
          "value": "63/66"
        },
        {
          "name": "M7 tool-call cases",
          "value": "20/22"
        }
      ],
      "details": "../../backend/20261026-201922/README.md",
      "_dir": "docs/benchmark/m5max-40c-128gb/qwen3.8_flash_next/oq4e/agent_ability/20261026-201922",
      "_details": "docs/benchmark/m5max-40c-128gb/qwen3.8_flash_next/oq4e/backend/20261026-201922/README.md",
      "_harness_source": "docs/benchmark/m5max-40c-128gb/qwen3.8_flash_next/oq4e/backend/20261026-201922/capability/results.json",
      "_harnesses": {
        "claude_code": {
          "gate": {
            "passed": 20,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 16.5,
            "modelCallsMedian": 6.0
          },
          "ranking": {
            "passed": 21,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 22.4,
            "modelCallsMedian": 6.0
          }
        },
        "hermes": {
          "gate": {
            "passed": 22,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 11.5,
            "modelCallsMedian": 5.0
          },
          "ranking": {
            "passed": 21,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 16.9,
            "modelCallsMedian": 4.0
          }
        },
        "localmelo": {
          "gate": {
            "passed": 22,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 9.8,
            "modelCallsMedian": 5.0
          },
          "ranking": {
            "passed": 21,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 15.4,
            "modelCallsMedian": 4.0
          }
        }
      }
    },
    {
      "schema": 1,
      "suite": "agent_ability",
      "protocol": "ui-real-1",
      "title": "UI on real web pages: 75 pages, 430 items x 3 samples; click, fill a form, read; from the screenshot or the accessibility list",
      "model": "Qwen3.8-Flash-Next oQ4e + MTP",
      "family": "qwen3.8_flash_next",
      "quant": "oq4e",
      "run_id": "20261027-ui-real",
      "date": "2026-10-27",
      "system": {
        "machine": "m5max-40c-128gb",
        "macos_build": "26B5101f",
        "xcode_build": "27A9275",
        "server": "oMLX 0.7.1rc1 (57620a3f)"
      },
      "metrics": [
        {
          "name": "Click, screenshot",
          "value": "545/594"
        },
        {
          "name": "Click, AX list",
          "value": "586/594"
        },
        {
          "name": "Fill, screenshot",
          "value": "36/72"
        },
        {
          "name": "Fill, AX list",
          "value": "45/72"
        },
        {
          "name": "Read, screenshot",
          "value": "541/624"
        },
        {
          "name": "Read, AX list",
          "value": "550/624"
        },
        {
          "name": "Read, original capture",
          "value": "534/624"
        }
      ],
      "notes": "Private set of replica pages, audited gold; reads by exact match. Screenshot + OCR, lenient reads and the failure analysis are in details.",
      "details": "../../../../qwen3.8_27b/oq5/agent_ability/20261027-ui-real/README.md",
      "_dir": "docs/benchmark/m5max-40c-128gb/qwen3.8_flash_next/oq4e/agent_ability/20261027-ui-real",
      "_details": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq5/agent_ability/20261027-ui-real/README.md",
      "_ui": {
        "replica:vision": {
          "click": [
            545,
            594
          ],
          "fill": [
            36,
            72
          ],
          "read_strict": [
            541,
            624
          ],
          "read_lenient": [
            572,
            624
          ]
        },
        "replica:vision+ocr": {
          "click": [
            549,
            594
          ],
          "fill": [
            41,
            72
          ],
          "read_strict": [
            518,
            624
          ],
          "read_lenient": [
            555,
            624
          ]
        },
        "replica:ax": {
          "click": [
            586,
            594
          ],
          "fill": [
            45,
            72
          ],
          "read_strict": [
            550,
            624
          ],
          "read_lenient": [
            571,
            624
          ]
        },
        "capture:vision": {
          "read_strict": [
            534,
            624
          ],
          "read_lenient": [
            568,
            624
          ]
        }
      },
      "_ui_source": "docs/benchmark/m5max-40c-128gb/qwen3.8_27b/oq5/agent_ability/20261027-ui-real/results.json",
      "_ui_meta": {
        "pages": 75,
        "items": {
          "click": 198,
          "fill": 24,
          "read": 208
        },
        "totalItems": 430,
        "samples": 3
      }
    },
    {
      "schema": 1,
      "suite": "agent_ability",
      "protocol": "capability-2",
      "title": "Claude Code, Hermes, localmelo on Aider 12 + MCPMark easy 10 (gate set) and Aider 22 (ranking set); thinking off",
      "model": "Qwen3.8-Flash-Next oQ4e + MTP",
      "family": "qwen3.8_flash_next",
      "quant": "oq4e",
      "run_id": "20261028-142438",
      "date": "2026-10-28",
      "system": {
        "machine": "m5max-40c-128gb",
        "macos_build": "26B123",
        "xcode_build": "27A9275",
        "server": "oMLX 0.7.1rc1 (7a5a3d75)"
      },
      "metrics": [
        {
          "name": "Claude Code (gate)",
          "value": "22/22"
        },
        {
          "name": "Hermes (gate)",
          "value": "22/22"
        },
        {
          "name": "localmelo (gate)",
          "value": "22/22"
        },
        {
          "name": "Gate set",
          "value": "66/66"
        },
        {
          "name": "Ranking set",
          "value": "62/66"
        },
        {
          "name": "M7 tool-call cases",
          "value": "\u2014"
        }
      ],
      "reference": true,
      "details": "../../backend/20261028-142438/capability/results.json",
      "_dir": "docs/benchmark/m5max-40c-128gb/qwen3.8_flash_next/oq4e/agent_ability/20261028-142438",
      "_details": "docs/benchmark/m5max-40c-128gb/qwen3.8_flash_next/oq4e/backend/20261028-142438/capability/results.json",
      "_harness_source": "docs/benchmark/m5max-40c-128gb/qwen3.8_flash_next/oq4e/backend/20261028-142438/capability/results.json",
      "_harnesses": {
        "claude_code": {
          "gate": {
            "passed": 22,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 17.2,
            "modelCallsMedian": 6.0
          },
          "ranking": {
            "passed": 20,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 32.2,
            "modelCallsMedian": 9.0
          }
        },
        "hermes": {
          "gate": {
            "passed": 22,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 12.6,
            "modelCallsMedian": 5.0
          },
          "ranking": {
            "passed": 22,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 17.2,
            "modelCallsMedian": 4.5
          }
        },
        "localmelo": {
          "gate": {
            "passed": 22,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 10.7,
            "modelCallsMedian": 5.0
          },
          "ranking": {
            "passed": 20,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 17.3,
            "modelCallsMedian": 5.5
          }
        }
      }
    },
    {
      "schema": 1,
      "suite": "agent_ability",
      "protocol": "capability-3",
      "title": "Claude Code, Hermes, localmelo on Aider 12 + MCPMark easy 10 (gate set) and Aider 22 (ranking set); thinking off",
      "model": "Qwen3.8-Flash-Next oQ4e + MTP",
      "family": "qwen3.8_flash_next",
      "quant": "oq4e",
      "run_id": "20261030-043547",
      "date": "2026-10-30",
      "system": {
        "machine": "m5max-40c-128gb",
        "macos_build": "26B123",
        "xcode_build": "27A9275",
        "server": "oMLX 0.7.1rc1 (7a5a3d75)"
      },
      "metrics": [
        {
          "name": "Claude Code (gate)",
          "value": "21/22"
        },
        {
          "name": "Hermes (gate)",
          "value": "22/22"
        },
        {
          "name": "localmelo (gate)",
          "value": "21/22"
        },
        {
          "name": "Gate set",
          "value": "64/66"
        },
        {
          "name": "Ranking set",
          "value": "61/66"
        },
        {
          "name": "M7 tool-call cases",
          "value": "\u2014"
        }
      ],
      "reference": true,
      "notes": "Xcode was updated from 27.1 (27A9275) to 27.2 (27B52) during the run; the capability tasks use no Xcode tooling beyond git.",
      "details": "../../backend/20261030-043547/capability/results.json",
      "_dir": "docs/benchmark/m5max-40c-128gb/qwen3.8_flash_next/oq4e/agent_ability/20261030-043547",
      "_details": "docs/benchmark/m5max-40c-128gb/qwen3.8_flash_next/oq4e/backend/20261030-043547/capability/results.json",
      "_harness_source": "docs/benchmark/m5max-40c-128gb/qwen3.8_flash_next/oq4e/backend/20261030-043547/capability/results.json",
      "_harnesses": {
        "claude_code": {
          "gate": {
            "passed": 21,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 19.0,
            "modelCallsMedian": 7.5
          },
          "ranking": {
            "passed": 20,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 27.8,
            "modelCallsMedian": 8.5
          }
        },
        "hermes": {
          "gate": {
            "passed": 22,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 11.9,
            "modelCallsMedian": 5.0
          },
          "ranking": {
            "passed": 21,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 16.7,
            "modelCallsMedian": 4.5
          }
        },
        "localmelo": {
          "gate": {
            "passed": 21,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 12.0,
            "modelCallsMedian": 5.0
          },
          "ranking": {
            "passed": 20,
            "runs": 22,
            "expected": 22,
            "modelTimeMedian": 17.9,
            "modelCallsMedian": 6.0
          }
        }
      }
    },
    {
      "schema": 1,
      "suite": "backend",
      "protocol": "speed-1",
      "title": "Speed on oMLX 0.7.1rc1 (M5 Max): text-only, agent sampling, thinking off",
      "model": "Qwen3.8-Flash-Next oQ4e + MTP",
      "family": "qwen3.8_flash_next",
      "quant": "oq4e",
      "run_id": "20261026-201922",
      "date": "2026-10-26",
      "system": {
        "machine": "m5max-40c-128gb",
        "macos_build": "26B5101f",
        "xcode_build": "27A9275",
        "server": "oMLX 0.7.1rc1 (7a5a3d75)"
      },
      "metrics": [
        {
          "name": "64K cold TTFT",
          "value": "31.5 s"
        },
        {
          "name": "Cached-prefix TTFT",
          "value": "1.27 s"
        },
        {
          "name": "Decode 16K",
          "value": "85.6 tok/s"
        },
        {
          "name": "Decode 32K",
          "value": "78.2 tok/s"
        },
        {
          "name": "Decode 64K",
          "value": "74.5 tok/s"
        },
        {
          "name": "Decode 110K",
          "value": "80.1 tok/s"
        },
        {
          "name": "20-step agent trajectory",
          "value": "61.8 s"
        },
        {
          "name": "Peak server memory",
          "value": "97.0 GiB"
        }
      ],
      "notes": "Kept with a minor disturbance flag (eval/backend/DESIGN.md \u00a75.4): 64K cold TTFT, Decode 16K, Decode 32K, Decode 64K, Decode 110K, Decode 16K with thinking on (in RESULTS.md, not in this table).",
      "details": "README.md",
      "_dir": "docs/benchmark/m5max-40c-128gb/qwen3.8_flash_next/oq4e/backend/20261026-201922",
      "_details": "docs/benchmark/m5max-40c-128gb/qwen3.8_flash_next/oq4e/backend/20261026-201922/README.md"
    }
  ],
  "taskSelection": {
    "gate": [
      "aider/affine-cipher",
      "aider/grade-school",
      "aider/list-ops",
      "aider/phone-number",
      "aider/pig-latin",
      "aider/proverb",
      "aider/react",
      "aider/robot-name",
      "aider/scale-generator",
      "aider/transpose",
      "aider/variable-length-quantity",
      "aider/wordy",
      "mcpmark/file_context/file_splitting",
      "mcpmark/file_context/pattern_matching",
      "mcpmark/file_context/uppercase",
      "mcpmark/file_property/largest_rename",
      "mcpmark/file_property/txt_merging",
      "mcpmark/folder_structure/structure_analysis",
      "mcpmark/legal_document/file_reorganize",
      "mcpmark/papers/papers_counting",
      "mcpmark/student_database/duplicate_name",
      "mcpmark/student_database/recommender_name"
    ],
    "ranking": [
      "aider/beer-song",
      "aider/book-store",
      "aider/bottle-song",
      "aider/bowling",
      "aider/connect",
      "aider/dominoes",
      "aider/dot-dsl",
      "aider/food-chain",
      "aider/forth",
      "aider/go-counting",
      "aider/grep",
      "aider/hangman",
      "aider/paasio",
      "aider/poker",
      "aider/pov",
      "aider/rest-api",
      "aider/sgf-parsing",
      "aider/simple-linked-list",
      "aider/tree-building",
      "aider/two-bucket",
      "aider/zebra-puzzle",
      "aider/zipper"
    ]
  },
  "sourceDigest": "dbac9739d776a0cf88018842e5e66e6ad3a12a6c144c1593600efcce5a1bbf4c"
};
