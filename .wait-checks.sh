#!/usr/bin/env bash
# Wait until no check is pending, then print the final check list.
PR=$1
REPO=$2
for _ in $(seq 1 40); do
  out=$(gh pr checks "$PR" -R "$REPO" 2>&1)
  if ! printf '%s' "$out" | grep -q pending; then
    echo "$out"
    exit 0
  fi
  sleep 10
done
echo "$out"
