use clap::Parser;
use dyno_term::cli::{parse_resolution, Args};

#[test]
fn test_cli_default_args() {
    let args = Args::try_parse_from(["dyno_term"]).expect("Failed to parse default CLI args");
    assert!(!args.benchmark);
    assert!(!args.demo);
    assert!(!args.test);
    assert!(!args.uncapped);
    assert_eq!(args.frames, 100);
    assert_eq!(args.res, "4k");
    assert_eq!(args.scene, 0);
    assert_eq!(args.output, "dyno_benchmark_results.json");
}

#[test]
fn test_cli_self_test_flag() {
    let args = Args::try_parse_from(["dyno_term", "--test"]).expect("Failed to parse --test");
    assert!(args.test);

    let args_alias =
        Args::try_parse_from(["dyno_term", "--self-test"]).expect("Failed to parse --self-test");
    assert!(args_alias.test);

    let args_short = Args::try_parse_from(["dyno_term", "-t"]).expect("Failed to parse -t");
    assert!(args_short.test);
}

#[test]
fn test_cli_benchmark_flags() {
    let args = Args::try_parse_from([
        "dyno_term",
        "--benchmark",
        "--frames",
        "250",
        "--res",
        "8k",
        "--output",
        "my_results.json",
    ])
    .expect("Failed to parse benchmark flags");

    assert!(args.benchmark);
    assert_eq!(args.frames, 250);
    assert_eq!(args.res, "8k");
    assert_eq!(args.output, "my_results.json");
}

#[test]
fn test_cli_history_and_baseline_flags() {
    let args_hist =
        Args::try_parse_from(["dyno_term", "--history"]).expect("Failed to parse --history");
    assert!(args_hist.history);

    let args_set_base_default =
        Args::try_parse_from(["dyno_term", "--set-baseline"]).expect("Failed to parse --set-baseline");
    assert_eq!(
        args_set_base_default.set_baseline,
        Some("dyno_benchmark_results.json".to_string())
    );

    let args_set_base_custom =
        Args::try_parse_from(["dyno_term", "--set-baseline", "custom_report.json"])
            .expect("Failed to parse --set-baseline custom");
    assert_eq!(
        args_set_base_custom.set_baseline,
        Some("custom_report.json".to_string())
    );

    let args_compare =
        Args::try_parse_from(["dyno_term", "--compare", "new_report.json", "--baseline", "base.json"])
            .expect("Failed to parse --compare");
    assert_eq!(args_compare.compare, Some("new_report.json".to_string()));
    assert_eq!(args_compare.baseline, Some("base.json".to_string()));
}

#[test]
fn test_cli_consistency_flags() {
    let args_default = Args::try_parse_from(["dyno_term", "--consistency"])
        .expect("Failed to parse --consistency default");
    assert_eq!(args_default.consistency, Some(5));
    assert_eq!(args_default.cooldown, 60);

    let args_short = Args::try_parse_from(["dyno_term", "-c", "3", "--cooldown", "30"])
        .expect("Failed to parse -c 3");
    assert_eq!(args_short.consistency, Some(3));
    assert_eq!(args_short.cooldown, 30);

    let args_alias = Args::try_parse_from(["dyno_term", "--stability", "10"])
        .expect("Failed to parse --stability");
    assert_eq!(args_alias.consistency, Some(10));
}

#[test]
fn test_cli_demo_flags() {
    let args = Args::try_parse_from(["dyno_term", "-d", "-u", "-s", "3", "--res", "1440p"])
        .expect("Failed to parse demo flags");

    assert!(args.demo);
    assert!(args.uncapped);
    assert_eq!(args.scene, 3);
    assert_eq!(args.res, "1440p");
}

#[test]
fn test_cli_gpu_selection_flags() {
    let args = Args::try_parse_from(["dyno_term", "--gpu", "1"]).expect("Failed to parse --gpu");
    assert_eq!(args.gpu, Some(1));

    let args_list =
        Args::try_parse_from(["dyno_term", "--list-gpus"]).expect("Failed to parse --list-gpus");
    assert!(args_list.list_gpus);
}

#[test]
fn test_cli_custom_dimensions() {
    let args = Args::try_parse_from(["dyno_term", "--width", "160", "--height", "80"])
        .expect("Failed to parse custom dimensions");
    assert_eq!(args.width, Some(160));
    assert_eq!(args.height, Some(80));
}

#[test]
fn test_parse_resolution_all_presets() {
    assert_eq!(parse_resolution("16k", 100, 100), (15360, 8640));
    assert_eq!(parse_resolution("8k", 100, 100), (7680, 4320));
    assert_eq!(parse_resolution("4k", 100, 100), (3840, 2160));
    assert_eq!(parse_resolution("2160p", 100, 100), (3840, 2160));
    assert_eq!(parse_resolution("uhd", 100, 100), (3840, 2160));
    assert_eq!(parse_resolution("1440p", 100, 100), (2560, 1440));
    assert_eq!(parse_resolution("2k", 100, 100), (2560, 1440));
    assert_eq!(parse_resolution("qhd", 100, 100), (2560, 1440));
    assert_eq!(parse_resolution("1080p", 100, 100), (1920, 1080));
    assert_eq!(parse_resolution("fhd", 100, 100), (1920, 1080));
    assert_eq!(parse_resolution("720p", 100, 100), (1280, 720));
    assert_eq!(parse_resolution("hd", 100, 100), (1280, 720));
    assert_eq!(parse_resolution("native", 120, 60), (120, 60));
    // Unknown fallback defaults to 4K UHD
    assert_eq!(parse_resolution("unknown_preset", 100, 100), (3840, 2160));
}

#[test]
fn test_cli_failsafe_flags() {
    let args = Args::try_parse_from([
        "dyno_term",
        "--max-temp",
        "98.5",
        "--no-failsafe",
        "--target-temp",
        "50.0",
    ])
    .expect("Failed to parse failsafe flags");
    assert_eq!(args.max_temp, 98.5);
    assert!(args.no_failsafe);
    assert_eq!(args.target_temp, 50.0);

    let args_default = Args::try_parse_from(["dyno_term"]).expect("Failed default parse");
    assert_eq!(args_default.max_temp, 105.0);
    assert!(!args_default.no_failsafe);
    assert_eq!(args_default.target_temp, 55.0);
}
