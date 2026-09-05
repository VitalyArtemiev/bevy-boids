#!/bin/sh
# Guards the invariant from SKILL.md hard constraint #7: the wasm-bindgen
# version in Cargo.lock must equal the wasm-bindgen-cli pin in
# .github/workflows/ci.yaml. If they drift apart, CI builds the wasm against
# one schema version and binds it with another, and the Pages deploy dies with
# "bindgen format ... must exactly match" after ~15 minutes of building.
# Used by .githooks/pre-commit (wire once: git config core.hooksPath .githooks)
# and as the first step of CI's check job. POSIX sh only (runs under git-bash
# on Windows dev boxes too).
set -eu

root="$(git rev-parse --show-toplevel)"
lock="$root/Cargo.lock"
wf="$root/.github/workflows/ci.yaml"

fail() {
    printf 'ERROR: %s\n\n' "$1" >&2
    exit 1
}

[ -f "$lock" ] || fail "Cargo.lock is missing — it must be committed so CI resolves identical dependency versions (a drifted wasm-bindgen once broke the Pages deploy)."
[ -f "$wf" ] || fail "could not find .github/workflows/ci.yaml"

# Regression guard for the original break: the lockfile must never be
# gitignored again, or CI floats dependency versions on every deploy.
if git -C "$root" check-ignore -q Cargo.lock; then
    fail "Cargo.lock is gitignored (.gitignore) — CI would resolve fresh versions and float past the wasm-bindgen-cli pin. Un-ignore it and commit it."
fi

# wasm-bindgen moves as a lockstep family (wasm-bindgen-futures, js-sys,
# web-sys exact-pin each other), so the crate version alone identifies the
# schema. Exact-match the name so wasm-bindgen-futures etc. don't shadow it.
lock_version="$(awk '
    $1 == "name" && $3 == "\"wasm-bindgen\"" { grab = 1; next }
    grab && $1 == "version" { gsub(/"/, "", $3); print $3; exit }
' "$lock")"
[ -n "$lock_version" ] || fail "could not find the wasm-bindgen version in Cargo.lock"

ci_version="$(sed -n 's/.*wasm-bindgen-cli --version \([0-9.]*\).*/\1/p' "$wf" | head -n1)"
[ -n "$ci_version" ] || fail "could not find the wasm-bindgen-cli --version pin in .github/workflows/ci.yaml"

if [ "$lock_version" != "$ci_version" ]; then
    {
        printf 'ERROR: wasm-bindgen version mismatch (the deploy-breaking kind):\n'
        printf '  Cargo.lock (crate linked into the wasm):  %s\n' "$lock_version"
        printf '  ci.yaml (wasm-bindgen-cli that binds it): %s\n' "$ci_version"
        printf 'Align them: update the pin in ci.yaml and/or move the lockfile with\n'
        printf '  cargo update -p wasm-bindgen -p wasm-bindgen-futures -p js-sys -p web-sys\n'
        printf '(they exact-pin each other and must move together), then commit both files.\n'
    } >&2
    exit 1
fi

# Non-blocking: catch a stale *local* CLI before a local web build hits the
# same schema error. Machines that never build for web can ignore this.
if command -v wasm-bindgen >/dev/null 2>&1; then
    local_version="$(wasm-bindgen --version | awk '{print $2}')"
    if [ "$local_version" != "$lock_version" ]; then
        printf 'WARNING: installed wasm-bindgen-cli is %s but the project links %s; local web builds will fail the schema check.\n' \
            "$local_version" "$lock_version" >&2
        printf '         Fix: cargo install wasm-bindgen-cli --version %s --locked\n' "$lock_version" >&2
    fi
fi

printf 'wasm-bindgen pin ok: %s (Cargo.lock == ci.yaml)\n' "$lock_version"
