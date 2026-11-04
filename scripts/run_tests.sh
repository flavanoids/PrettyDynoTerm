#!/usr/bin/env bash
set -e

echo -e "\x1b[38;2;0;255;200;1m================================================================================"
echo -e "                   DYNO_TERM COMPREHENSIVE TEST RUNNER"
echo -e "================================================================================\x1b[0m"

echo -e "\n\x1b[38;2;0;220;255;1m[1/2] Running Rust Unit & Integration Test Suites (cargo test)...\x1b[0m"
cargo test --all-targets -- --nocapture

echo -e "\n\x1b[38;2;0;220;255;1m[2/2] Running CLI Binary Self-Test & Hardware Verification (dyno_term --test)...\x1b[0m"
cargo run -- --test

echo -e "\n\x1b[38;2;0;255;128;1m================================================================================"
echo -e "  ALL COMPREHENSIVE TEST SUITES COMPLETED SUCCESSFULLY"
echo -e "================================================================================\x1b[0m"
