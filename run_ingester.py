#!/usr/bin/env python3
"""
run_ingester.py — build and launch the Rust Bybit market-data ingester.

Usage
-----
    python run_ingester.py [--debug]

How it works
------------
1. Checks whether `cargo` is on PATH.
   If not, installs the Rust toolchain via rustup (non-interactive).
2. Runs `cargo build --release` inside bybit_ingester/.
3. Replaces the current Python process with the compiled binary via
   os.execvp — zero wrapper overhead at runtime, clean signal handling.

Railway usage
-------------
Set the bybit-ingester service start command to:
    python run_ingester.py
and leave the Dockerfile as the standard Python one (Dockerfile).
No separate Dockerfile.rust or Rust image is required.

Environment variables forwarded to the binary
----------------------------------------------
All environment variables present when this script is called are
inherited by the binary (os.execvp preserves the environment).
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] run_ingester — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
INGESTER_DIR = os.path.join(SCRIPT_DIR, "bybit_ingester")
BINARY_PATH  = os.path.join(INGESTER_DIR, "target", "release", "bybit_ingester")
RUSTUP_INIT_URL = "https://sh.rustup.rs"


# ── helpers ──────────────────────────────────────────────────────────────────

def _run(args: list[str], cwd: str | None = None) -> None:
    """Run a command, streaming output, and raise on non-zero exit."""
    log.info("$ %s", " ".join(args))
    result = subprocess.run(args, cwd=cwd)
    if result.returncode != 0:
        log.error("Command failed with exit code %d", result.returncode)
        sys.exit(result.returncode)


def ensure_cargo() -> str:
    """Return path to cargo, installing rustup/cargo if necessary."""
    cargo = shutil.which("cargo")
    if cargo:
        log.info("cargo found: %s", cargo)
        return cargo

    # cargo not on PATH — check the default rustup install location
    home_cargo = os.path.expanduser("~/.cargo/bin/cargo")
    if os.path.isfile(home_cargo):
        log.info("cargo found at %s (not on PATH)", home_cargo)
        _add_cargo_to_path()
        return home_cargo

    log.info("Rust toolchain not found — installing via rustup ...")
    _install_rustup()

    home_cargo = os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.isfile(home_cargo):
        log.error("rustup install appeared to succeed but cargo is still missing at %s", home_cargo)
        sys.exit(1)

    _add_cargo_to_path()
    return home_cargo


def _add_cargo_to_path() -> None:
    """Prepend ~/.cargo/bin to PATH for this process and all children."""
    cargo_bin = os.path.expanduser("~/.cargo/bin")
    current = os.environ.get("PATH", "")
    if cargo_bin not in current.split(os.pathsep):
        os.environ["PATH"] = cargo_bin + os.pathsep + current
        log.info("Added %s to PATH", cargo_bin)


def _install_rustup() -> None:
    """Download and run the rustup init script non-interactively."""
    # curl must be present (standard on every Railway/Debian image).
    curl = shutil.which("curl")
    if not curl:
        log.error("curl is required to install rustup but was not found on PATH")
        sys.exit(1)

    # -s silent, -S show errors, -f fail on HTTP error, -L follow redirects
    # --proto/--tlsv1.2 lock down the transport
    install_cmd = [
        curl, "-sSfL", "--proto", "=https", "--tlsv1.2",
        RUSTUP_INIT_URL, "-o", "/tmp/rustup-init.sh",
    ]
    _run(install_cmd)

    os.chmod("/tmp/rustup-init.sh", 0o755)
    # -y  non-interactive, --no-modify-path  we handle PATH ourselves
    _run(["sh", "/tmp/rustup-init.sh", "-y", "--no-modify-path"])
    log.info("Rust toolchain installed successfully")


# ── build ─────────────────────────────────────────────────────────────────────

def build(debug: bool = False) -> None:
    """Compile the Rust crate."""
    if not os.path.isdir(INGESTER_DIR):
        log.error("bybit_ingester/ directory not found at %s", INGESTER_DIR)
        sys.exit(1)

    cargo = ensure_cargo()
    args = [cargo, "build"]
    if not debug:
        args.append("--release")

    log.info("Building bybit_ingester (%s) ...", "debug" if debug else "release")
    _run(args, cwd=INGESTER_DIR)
    log.info("Build complete → %s", BINARY_PATH)


# ── launch ────────────────────────────────────────────────────────────────────

def launch(debug: bool = False) -> None:
    """exec() the compiled binary — replaces this Python process."""
    binary = BINARY_PATH
    if debug:
        binary = os.path.join(INGESTER_DIR, "target", "debug", "bybit_ingester")

    if not os.path.isfile(binary):
        log.error("Binary not found at %s — build must have failed", binary)
        sys.exit(1)

    log.info("Launching %s ...", binary)
    os.execvp(binary, [binary])   # replaces this process; never returns


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build (if needed) and run the Rust Bybit ingester."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Build and run a debug (unoptimised) binary instead of release.",
    )
    args = parser.parse_args()

    build(debug=args.debug)
    launch(debug=args.debug)


if __name__ == "__main__":
    main()
