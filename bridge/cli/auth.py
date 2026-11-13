"""CLI command handler for provider authentication (`abridge auth aistudio`)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import getpass
import sys
from typing import Any

from bridge.aistudio_auth import (
    AistudioKeychainProvider,
    check_aistudio_auth_health,
    redact_secret,
)
from bridge.errors import AuthenticationError


def _verify_key_if_requested(
    api_key: str,
    verifier: Callable[[str], tuple[str, str]] | None,
) -> tuple[str, str]:
    if verifier is not None:
        return verifier(api_key)
    try:
        from bridge.aistudio_client import verify_aistudio_api_key
    except ImportError:
        return ("configured_unverified", "Stored in Keychain (live check unavailable).")
    return verify_aistudio_api_key(api_key)


def _handle_auth_aistudio(
    argv: list[str],
    prog_base: str,
    provider: AistudioKeychainProvider | None = None,
    verifier: Callable[[str], tuple[str, str]] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog=f"{prog_base} auth aistudio",
        description=(
            "Configure, inspect, or remove the Google AI Studio Gemini API key "
            "stored securely in macOS Keychain (never saved in config files or argv)."
        ),
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=["set", "status", "remove"],
        default="set",
        help="Action to perform (default: set)",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read the Gemini API key from stdin instead of an interactive hidden prompt.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Save the API key to Keychain without making a live probe request to Google.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="When checking status, perform a live check against Google AI Studio.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Shortcut for `auth aistudio status`.",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Shortcut for `auth aistudio remove`.",
    )

    args = parser.parse_args(argv)
    action = args.action
    if args.status:
        action = "status"
    elif args.remove:
        action = "remove"

    prov = provider if provider is not None else AistudioKeychainProvider()

    if action == "status":
        health = check_aistudio_auth_health(
            provider=prov,
            verifier=verifier,
            verify_live=bool(args.verify),
        )
        print(f"AI Studio Keychain item: service={health['service']} account={health['account']}")
        print(f"Status: {health['status']} — {health['detail']}")
        return 0 if health["configured"] else 1

    if action == "remove":
        try:
            removed = prov.delete_api_key()
        except AuthenticationError as exc:
            print(f"Error: {redact_secret(str(exc))}", file=sys.stderr)
            return 1
        if removed:
            print(f"Removed AI Studio API key from Keychain (service={prov.service}, account={prov.account}).")
        else:
            print(f"No AI Studio API key was stored in Keychain (service={prov.service}, account={prov.account}).")
        return 0

    # action == "set"
    if args.stdin:
        secret = sys.stdin.readline().strip()
    else:
        print("Create a Gemini API key at: https://aistudio.google.com/api-keys")
        print(f"Target macOS Keychain item: service={prov.service}, account={prov.account}")
        secret = getpass.getpass("Enter Google AI Studio Gemini API key (hidden): ").strip()

    if not secret:
        print("Error: Gemini API key cannot be empty.", file=sys.stderr)
        return 1

    try:
        prov.store_api_key(secret)
    except (ValueError, AuthenticationError) as exc:
        print(f"Error: {redact_secret(str(exc), secret)}", file=sys.stderr)
        return 1

    print(f"Saved AI Studio Gemini API key in macOS Keychain (service={prov.service}, account={prov.account}).")
    if args.skip_verify:
        return 0

    try:
        v_status, v_detail = _verify_key_if_requested(secret, verifier)
    except Exception as exc:
        print(f"Status: unverified — {redact_secret(str(exc), secret)}")
        return 0

    print(f"Status: {v_status} — {redact_secret(v_detail, secret)}")
    if v_status == "rejected":
        print(
            "Warning: Google rejected this API key. Check permissions or replace it via `abridge auth aistudio`.",
            file=sys.stderr,
        )
        return 1
    return 0


def handle_auth(
    argv: list[str],
    prog_base: str,
    provider: Any | None = None,
    verifier: Callable[[str], tuple[str, str]] | None = None,
) -> int:
    """Dispatches `abridge auth <provider> ...` subcommands."""
    if not argv or argv[0] in ("-h", "--help", "-help"):
        print(f"usage: {prog_base} auth <provider> [set|status|remove] [options]\n")
        print("Supported providers:")
        print("  aistudio    Manage Google AI Studio Gemini API key in macOS Keychain")
        return 0

    target = argv[0].strip().lower()
    if target == "aistudio":
        return _handle_auth_aistudio(argv[1:], prog_base, provider=provider, verifier=verifier)

    print(f"Error: Unknown auth provider '{argv[0]}'. Supported: aistudio.", file=sys.stderr)
    return 2
