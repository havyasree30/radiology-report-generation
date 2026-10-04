"""F1 step 2: the ONE-SHOT gate that opens the locked test partition.

Preconditions (all checked; any failure stops before the marker is written):
  * the F1 protocol JSON / MD still match their recorded SHA-256 (f1_01);
  * the freeze verification is on file and passed;
  * the validation dry run reproduced the stored R1 / R2 / G1 / G1F results (f1_03 and f1_05 on validation);
  * the test is not already open.
Actions: hash every F1 analysis script (code freeze, written BEFORE any test content is read) and write `test/test_opened.json`. From this moment on
only the frozen code may touch the test partition; the marker cannot be written twice.

    .venv\\Scripts\\python.exe -m scripts.f1_02_open_test
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone

from src.classification.final_test import sha256_file
from src.f1.pipeline import F1
from src.utils.config import PROJECT_ROOT


def main() -> int:
    marker = F1 / "test/test_opened.json"
    if marker.exists():
        print("STOP: the locked test was already opened; it cannot be opened a second time", file=sys.stderr)
        return 1
    rec = {ln.split(" *")[1]: ln.split(" *")[0] for ln in (F1 / "F1_LOCKED_TEST_PROTOCOL.sha256").read_text(encoding="utf-8").splitlines() if ln.strip()}
    ver = json.loads((F1 / "f1_freeze_verification.json").read_text(encoding="utf-8"))
    dry = json.loads((F1 / "validation_dry_run/reproduction_checks.json").read_text(encoding="utf-8"))
    dry_e = json.loads((F1 / "validation_dry_run/evaluation_reproduction_checks.json").read_text(encoding="utf-8"))
    gen_dry = json.loads((F1 / "validation_dry_run/generation_dry_run_check.json").read_text(encoding="utf-8"))
    checks = {"protocol_json_hash_unchanged": sha256_file(F1 / "F1_LOCKED_TEST_PROTOCOL.json") == rec["F1_LOCKED_TEST_PROTOCOL.json"], "protocol_md_hash_unchanged": sha256_file(F1 / "F1_LOCKED_TEST_PROTOCOL.md") == rec["F1_LOCKED_TEST_PROTOCOL.md"],
              "freeze_verification_passed": bool(ver["all_identifiers_verified"]), "dry_run_prepare_reproduced": bool(dry["all_reproduced"]), "dry_run_evaluation_reproduced": bool(dry_e["all_reproduced"]),
              "dry_run_generation_path_ran_without_failure": bool(gen_dry["log"]["n_failed"] == 0)}
    if not all(checks.values()):
        print(json.dumps({"STOP": "preconditions failed; the locked test is NOT opened", "checks": checks}, indent=1), file=sys.stderr)
        return 1
    code = sorted((PROJECT_ROOT / "src/f1").glob("*.py")) + [p for p in sorted((PROJECT_ROOT / "scripts").glob("f1_0[1-6]_*.py"))]
    hashes = {str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(p) for p in code}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    freeze = {"frozen_utc": now, "scope": "analysis code that computes every scientific metric (population, classifier run, routing, retrieval, generation, evaluation, comparison, opening gate)", "sha256": hashes,
              "combined_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), "protocol_json_sha256": rec["F1_LOCKED_TEST_PROTOCOL.json"], "preconditions": checks,
              "not_in_scope": "figure, table, document, integrity and test scripts only format stored values; they are written/changed after the run and listed in the final integrity report",
              "working_tree_note": "F1 code and protocol are uncommitted at this point; file hashes are the record"}
    (F1 / "f1_code_freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    (F1 / "test").mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"opened_utc": now, "protocol_json_sha256": rec["F1_LOCKED_TEST_PROTOCOL.json"], "code_freeze_combined_sha256": freeze["combined_sha256"], "note": "written once; the locked test partition is open from this moment"}, indent=2), encoding="utf-8")
    print(json.dumps({"opened_utc": now, "code_freeze_combined_sha256": freeze["combined_sha256"], "n_files_frozen": len(hashes)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
