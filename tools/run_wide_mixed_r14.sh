#!/usr/bin/env bash
# Run from the Ubuntu study root after staging r14/src and the launching helpers.
set -euo pipefail
study=wide-mixed-r14
run_helper() {
    python3 run_isolated_perception_job.py --helper "$1" --label "$2" --seconds "$3" --minimum-free-gib 1 "${@:4}"
}

run_helper wide_mixed_r14_checks.py wide-r14-controller-tests 360
cp wide-r14-controller-tests/run.log "$study/controller-tests.log"
run_helper prepare_wide_mixed_r14.py wide-r14-prepare 300
run_helper preflight_sized_autonomy.py wide-r14-geometry 5400 -- --study "$study"
python3 run_sized_autonomy.py --study "$study" --part development --label wide-r14-dev --seconds 7200
python3 compress_sized_recordings.py --study "$study" --part development --release-verified-originals
for shard in {1..10}; do
    python3 run_sized_autonomy.py --study "$study" --part test --shard "$shard" --label "wide-r14-test-$shard" --seconds 7200
    python3 compress_sized_recordings.py --study "$study" --part test --shard "$shard" --release-verified-originals
done
python3 close_wide_mixed_autonomy.py --study "$study"
