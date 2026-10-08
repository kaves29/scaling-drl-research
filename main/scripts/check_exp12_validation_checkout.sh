#!/usr/bin/env bash
# Source from a Slurm wrapper after entering its explicit submission directory.
exp12_checkout_fail() {
  printf 'exp12 validation: preflight failed: %s\n' "$*" >&2
  exit 2
}

exp12_check_checkout() {
  [ -n "${EXPECTED_COMMIT:-}" ] || exp12_checkout_fail "EXPECTED_COMMIT is required"
  [[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exp12_checkout_fail "EXPECTED_COMMIT must be a full lowercase 40-character hash"
  local actual path status root
  actual=$(git rev-parse --verify HEAD) || exp12_checkout_fail "cannot resolve Git HEAD"
  [ "$actual" = "$EXPECTED_COMMIT" ] || exp12_checkout_fail "HEAD mismatch: expected $EXPECTED_COMMIT, found $actual"
  git diff --quiet && git diff --cached --quiet || exp12_checkout_fail "tracked or staged working-tree changes"
  # These existing validation drivers write their artifacts under main/logs/.
  # Inspect untracked source separately so Slurm's already-opened stderr file
  # does not reject an otherwise clean checkout. Tracked logs remain checked.
  root=$(git rev-parse --show-toplevel) || exp12_checkout_fail "cannot resolve checkout root"
  status=$(git -C "$root" ls-files --others --exclude-standard -- . ':(exclude)main/logs') || exp12_checkout_fail "cannot inspect untracked source"
  [ -z "$status" ] || exp12_checkout_fail "untracked source files outside logs/"
  for path in "$@"; do
    [ -f "$path" ] || exp12_checkout_fail "missing required file: $path"
    git ls-files --error-unmatch "$path" >/dev/null 2>&1 || exp12_checkout_fail "required file is not tracked: $path"
  done
  [ -z "${BLOCKB_TEST_HOOKS:-}" ] || exp12_checkout_fail "BLOCKB_TEST_HOOKS is forbidden in a Slurm validation"
  printf 'Validated submission checkout: %s at %s\n' "$PWD" "$actual"
}
