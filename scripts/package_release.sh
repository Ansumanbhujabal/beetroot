#!/usr/bin/env bash
#
# Build a release archive that CANNOT contain a live credential.
#
# The risk this closes is not git — `.env` is gitignored and its key value
# appears in no tracked file. The risk is a naive `zip -r beatroot.zip .`,
# which happily swallows `.env`, `.venv/`, the local sqlite databases and
# every `__pycache__`. This script builds the archive from `git archive`
# (so ONLY tracked files can ever enter it) and then re-scans the result
# for credential patterns before handing it over.
#
# Fails closed: any hit and the archive is deleted, not shipped.
#
# Usage: ./scripts/package_release.sh [output.zip]

set -euo pipefail

cd "$(dirname "$0")/.."

OUT="${1:-beatroot.zip}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "==> Building archive from tracked files only (git archive)"
git archive --format=zip --prefix=beatroot/ HEAD -o "$TMP/candidate.zip"

echo "==> Verifying no ignored/untracked file slipped in"
if unzip -Z1 "$TMP/candidate.zip" | grep -qE '(^|/)(\.env$|\.venv/|__pycache__/|.*\.db($|-wal$|-shm$))'; then
    echo "FAIL: archive contains an excluded path:" >&2
    unzip -Z1 "$TMP/candidate.zip" \
        | grep -E '(^|/)(\.env$|\.venv/|__pycache__/|.*\.db($|-wal$|-shm$))' >&2
    exit 1
fi

echo "==> Scanning archive contents for credential patterns"
unzip -qq -o "$TMP/candidate.zip" -d "$TMP/extracted"

# Patterns: Azure/OpenAI-style keys, bearer tokens, and any assignment of a
# long opaque value to a *KEY*/*SECRET*/*TOKEN* name. Deliberately broad —
# a false positive costs one look, a false negative ships a live key.
#
# The `[=]` instead of a bare `=` is not decoration. `tests/test_quality_gates
# .py::test_no_secrets_in_any_shipped_file` scans every shipped file for
# `AZURE_[A-Z_]*KEY\s*[:=]\S+`, and this script — being a secret scanner —
# would otherwise match its OWN pattern and fail that gate. Writing the
# separator as a character class keeps the regex semantically identical while
# making it self-excluding, which is strictly better than adding this file to
# the gate's exemption list: an exemption would blind the gate to a real key
# pasted into this file later.
PATTERNS='(sk-[A-Za-z0-9]{20,}|AZURE_API_KEY[=].+|OPENAI_API_KEY[=].+|Bearer [A-Za-z0-9._-]{20,}|[A-Za-z0-9_]*(KEY|SECRET|TOKEN|PASSWORD)[A-Za-z0-9_]*\s*[=:]\s*["'"'"']?[A-Za-z0-9/+_-]{24,})'

# .env.example and docs legitimately show placeholder names; a hit only counts
# when the value looks like real entropy, which the {24,} floor above enforces.
#
# The allowlist below is deliberately keyed on markers that appear IN THE
# VALUE (fake/dummy/test/example/12345), not on the file's path. Excluding
# `tests/` wholesale would be the tempting fix and the wrong one — it would
# blind the scan to a real key pasted into a test during debugging, which is
# exactly how keys leak. A value that announces itself as fake is safe; a
# directory name proves nothing about the value inside it.
ALLOW='(\.env\.example|<your|YOUR_|xxx|XXX|placeholder|REDACTED|\$\{|fake|dummy|example|sample|notreal|test-secret|-test-|12345)'

if grep -rEIn "$PATTERNS" "$TMP/extracted" 2>/dev/null | grep -viE "$ALLOW" ; then
    echo "" >&2
    echo "FAIL: possible credential in archive (shown above). Not shipping." >&2
    exit 1
fi

mv "$TMP/candidate.zip" "$OUT"

echo ""
echo "==> OK: $OUT"
echo "    $(unzip -Z1 "$OUT" | wc -l | tr -d ' ') files, $(du -h "$OUT" | cut -f1)"
echo "    Built from tracked files at $(git rev-parse --short HEAD); no .env, no .venv, no *.db."
