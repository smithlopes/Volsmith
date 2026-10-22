from __future__ import annotations

import subprocess
import sys
from tempfile import TemporaryFile
import time
from pathlib import Path


PLOTS_DIR = Path(__file__).resolve().parent
ASSET_CLASS_DIRS = (
    PLOTS_DIR / "Commodities Markets",
    PLOTS_DIR / "Equity Markets",
    PLOTS_DIR / "FX Markets",
)
EQUITY_START_DELAY_SECONDS = 5


def main() -> int:
    scripts = sorted(
        script
        for folder in ASSET_CLASS_DIRS
        for script in folder.glob("*.py")
        if script.name != "Breadth of Market Growth.py"
    )

    if not scripts:
        print("No asset-class scripts found.", file=sys.stderr)
        return 1

    processes: list[tuple[Path, subprocess.Popen, object]] = []
    equity_scripts_started = 0

    for script in scripts:
        is_equity_script = script.parent.name == "Equity Markets"
        if is_equity_script and equity_scripts_started:
            print(
                f"Waiting {EQUITY_START_DELAY_SECONDS} seconds before "
                "starting the next Equity script...",
                flush=True,
            )
            time.sleep(EQUITY_START_DELAY_SECONDS)

        print(f"Starting: {script.relative_to(PLOTS_DIR)}", flush=True)
        output_log = TemporaryFile(
            mode="w+t",
            encoding="utf-8",
            errors="replace",
        )
        process = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=script.parent,
            stdout=output_log,
            stderr=subprocess.STDOUT,
        )
        processes.append((script, process, output_log))
        if is_equity_script:
            equity_scripts_started += 1

    failed = False
    separator = "=" * 88
    for script, process, output_log in processes:
        return_code = process.wait()
        output_log.seek(0)
        script_output = output_log.read()
        relative_script = script.relative_to(PLOTS_DIR)

        print(f"\n{separator}\nConsole output: {relative_script}\n{separator}")
        if script_output:
            sys.stdout.write(script_output)
            if not script_output.endswith("\n"):
                print()
        print(separator, flush=True)
        output_log.close()

        if return_code:
            failed = True
            print(
                f"Failed ({return_code}): {relative_script}",
                file=sys.stderr,
            )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
