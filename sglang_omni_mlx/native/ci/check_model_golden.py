# SPDX-License-Identifier: Apache-2.0
"""Checks a native model's parity tool against its golden outputs on the frozen corpus.

    python check_model_golden.py --runtime-bin DIR --data-root DIR --golden FILE [--write]

The golden file names the parity tool (whisper_transcribe, ...), its request
flags, and the language each clip language is sent with (a user with that main
language). Every corpus clip is transcribed, one tool run per language sent,
and each clip's text, language, token count and finish reason must equal the
golden file. Word, character and mixed error rates are reported next to the
original Voxt backend's (Swift on MLX Audio) as a summary; they do not gate.
--write regenerates the golden clips and metrics from the current runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

from check_golden import CI_DIRECTORY, COMPARED_FIELDS, quality


class ClipResult(TypedDict):
    text: str
    language: str | None
    generated_token_count: int
    finish_reason: str


def transcribe(
    command: list[str], clips: list[Path], language: str | None
) -> dict[str, ClipResult]:
    language_flags = ["--language", language] if language is not None else []
    completed = subprocess.run(
        command + language_flags + [str(clip) for clip in clips],
        check=True,
        capture_output=True,
        text=True,
    )
    results = {}
    for line in completed.stdout.splitlines():
        row = json.loads(line)
        results[Path(row["file"]).stem] = {
            field: row[field] for field in COMPARED_FIELDS
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-bin", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()

    golden = json.loads(arguments.golden.read_text())
    manifest = {
        row["id"]: row
        for row in map(
            json.loads,
            (CI_DIRECTORY / "corpus" / "manifest.jsonl").read_text().splitlines(),
        )
    }
    model_directory = arguments.data_root / "models" / golden["model"].replace("/", "_")
    command = [
        str(arguments.runtime_bin / golden["tool"]),
        "--model-path",
        str(model_directory),
    ]
    for flag, value in golden["request"].items():
        command += [f"--{flag.replace('_', '-')}", str(value)]
    clips_by_request_language: dict[str | None, list[Path]] = {}
    for clip_id, clip in manifest.items():
        clips_by_request_language.setdefault(
            golden["language_by_clip_language"][clip["lang"]], []
        ).append(arguments.data_root / "corpus" / "v1" / "clips" / f"{clip_id}.wav")
    results: dict[str, ClipResult] = {}
    for language, clips in clips_by_request_language.items():
        results.update(transcribe(command, clips, language))
    results = {clip_id: results[clip_id] for clip_id in manifest}
    metrics = quality(
        manifest, {clip_id: row["text"] for clip_id, row in results.items()}
    )

    if arguments.write:
        golden["metrics"] = metrics
        golden["clips"] = results
        arguments.golden.write_text(
            json.dumps(golden, ensure_ascii=False, indent=1) + "\n"
        )
        print(f"wrote {len(results)} clips to {arguments.golden}")
        return
    else:
        pass

    mismatches = [
        clip_id for clip_id in manifest if results[clip_id] != golden["clips"][clip_id]
    ]
    baseline_pending = golden["baseline"]["source"] == "pending"
    lines = [
        f"### {golden['model']}",
        "",
        f"Golden clips identical: {len(manifest) - len(mismatches)}/{len(manifest)}",
        "",
    ]
    lines.extend(
        ["Swift baseline pending; native metrics are shown without a parity delta.", ""]
        if baseline_pending
        else []
    )
    lines.extend(["| | Original Voxt (Swift) | Native runtime | Δ |", "|---|---|---|---|"])
    for name, value in metrics.items():
        if baseline_pending:
            lines.append(f"| {name} | pending | {value:.2%} | n/a |")
        else:
            baseline = golden["baseline"][name]
            lines.append(
                f"| {name} | {baseline:.2%} | {value:.2%} | {(value - baseline) * 100:+.2f} pp |"
            )
    for clip_id in mismatches[:10]:
        lines.append(f"\n- `{clip_id}` differs from its golden output")
    report = "\n".join(lines) + "\n"
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write(report)
    else:
        pass
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
