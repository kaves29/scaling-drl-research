#!/bin/bash
# Packs a Block A or Block B output directory into one file to paste: the report (or summary), then for
# every step that did not pass (FAIL, TIMEOUT, exit-3 stop) the end of its log and every Python traceback.
#   bash scripts/collect_report.sh logs/blockB_<jobid>    # writes logs/blockB_<jobid>/paste_me.txt
set -u
dir=${1:?usage: $0 logs/blockB_<jobid> (or logs/blockA_<jobid>)}
out="$dir/paste_me.txt"

tracebacks() {  # every "Traceback ..." block through its exception line
  awk '/^Traceback \(most recent call last\):/ {p = 1}
       p {print}
       p && !/^Traceback/ && !/^[ \t]/ {print "----"; p = 0}' "$1" | head -n 400
}

failed_steps() {  # "<log path>\t<step>\t<result>" for every step that did not pass
  if [ -d "$dir/status" ]; then
    cat "$dir"/status/*.tsv 2>/dev/null | awk -F'\t' -v d="$dir" \
      '$3 !~ /^(PASS|SKIPPED)/ {print d "/" $1 "/" $2 ".log\t" $1 "/" $2 "\t" $3}'
  elif [ -f "$dir/status.tsv" ]; then
    awk -F'\t' -v d="$dir" 'NR > 1 && $2 !~ /^(PASS|SKIPPED)/ {print d "/" $1 ".log\t" $1 "\t" $2}' "$dir/status.tsv"
  fi
}

{
  echo "=== $(basename "$dir"), collected $(date -Is) ==="
  for f in report.txt summary.txt; do
    [ -f "$dir/$f" ] && { echo; echo "=== $f ==="; cat "$dir/$f"; }
  done
  [ -s "$dir/report_errors.log" ] && { echo; echo "=== report_errors.log ==="; tail -n 40 "$dir/report_errors.log"; }
  failed_steps | while IFS=$'\t' read -r log step result; do
    echo; echo "=== $step: $result ==="
    if [ -f "$log" ]; then
      echo "--- tracebacks ---"; tracebacks "$log"
      echo "--- last 80 lines of $log (Orbax deprecation warnings removed) ---"
      grep -v "SaveArgs.aggregate is deprecated" "$log" | tail -n 80
    else
      echo "(no log at $log)"
    fi
  done
  for lane in "$dir"/gpu*/lane.log; do
    [ -f "$lane" ] && { echo; echo "=== $lane (last 15 lines) ==="; tail -n 15 "$lane"; }
  done
} > "$out"
echo "$out ($(wc -c < "$out") bytes)"
