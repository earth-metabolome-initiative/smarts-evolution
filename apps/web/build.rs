//! Generates the dedicated wasm worker assets used by the SMARTS evolution UI.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

const WORKER_STEM: &str = "smarts_evolution_web_worker";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../Cargo.lock");
    println!("cargo:rerun-if-changed=../../src");
    println!("cargo:rerun-if-changed=../web-worker/Cargo.toml");
    println!("cargo:rerun-if-changed=../web-worker/src");
    println!("cargo:rerun-if-changed=../web-protocol/src");

    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR")
            .ok_or_else(|| "cargo did not provide CARGO_MANIFEST_DIR".to_owned())?,
    );
    let generated_dir = manifest_dir.join("public/generated");
    let worker_manifest = manifest_dir
        .join("../web-worker/Cargo.toml")
        .canonicalize()
        .map_err(|error| format!("failed to resolve worker manifest: {error}"))?;

    build_worker_assets(&worker_manifest, &generated_dir)
}

fn build_worker_assets(worker_manifest: &Path, generated_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(generated_dir)
        .map_err(|error| format!("failed to create generated worker directory: {error}"))?;

    let out_dir = PathBuf::from(
        env::var_os("OUT_DIR").ok_or_else(|| "cargo did not provide OUT_DIR".to_owned())?,
    );
    let bindgen_dir = out_dir.join("smarts-evolution-worker-bindgen");
    let target_dir = out_dir.join("smarts-evolution-worker-target");
    let _ignored = fs::remove_dir_all(&bindgen_dir);
    fs::create_dir_all(&bindgen_dir)
        .map_err(|error| format!("failed to create worker bindgen directory: {error}"))?;

    let cargo = env::var("CARGO").unwrap_or_else(|_| String::from("cargo"));
    let profile = env::var("PROFILE").unwrap_or_else(|_| String::from("debug"));

    let mut build = Command::new(cargo);
    let profile_debug_env = format!(
        "CARGO_PROFILE_{}_DEBUG",
        profile.replace('-', "_").to_ascii_uppercase()
    );
    build
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env(&profile_debug_env, "0")
        .env("CARGO_PROFILE_DEV_DEBUG", "0")
        .args(["build", "--manifest-path"])
        .arg(worker_manifest)
        .args([
            "--lib",
            "--target",
            "wasm32-unknown-unknown",
            "--target-dir",
        ])
        .arg(&target_dir);

    match profile.as_str() {
        "debug" => {}
        "release" => {
            build.arg("--release");
        }
        other => {
            build.args(["--profile", other]);
        }
    }

    let status = build
        .status()
        .map_err(|error| format!("failed to launch worker cargo build: {error}"))?;
    if !status.success() {
        return Err(format!("worker cargo build failed with status {status}"));
    }

    let worker_wasm = target_dir
        .join("wasm32-unknown-unknown")
        .join(&profile)
        .join(format!("{WORKER_STEM}.wasm"));
    if !worker_wasm.exists() {
        return Err(format!("expected worker wasm at {}", worker_wasm.display()));
    }

    let mut bindgen = wasm_bindgen_cli_support::Bindgen::new();
    bindgen
        .input_path(&worker_wasm)
        .out_name(WORKER_STEM)
        .typescript(false)
        .web(true)
        .map_err(|error| format!("failed to configure worker bindgen for web output: {error}"))?
        .generate(&bindgen_dir)
        .map_err(|error| format!("worker bindgen generation failed: {error}"))?;

    copy_file(
        &bindgen_dir.join(format!("{WORKER_STEM}.js")),
        &generated_dir.join(format!("{WORKER_STEM}.js")),
    )?;
    copy_file(
        &bindgen_dir.join(format!("{WORKER_STEM}_bg.wasm")),
        &generated_dir.join(format!("{WORKER_STEM}_bg.wasm")),
    )?;
    fs::write(
        generated_dir.join("evolution-worker.js"),
        worker_loader_script(),
    )
    .map_err(|error| format!("failed to write worker bootstrap script: {error}"))?;
    Ok(())
}

fn copy_file(source: &Path, destination: &Path) -> Result<(), String> {
    fs::copy(source, destination).map_err(|error| {
        format!(
            "failed to copy generated worker asset from {} to {}: {error}",
            source.display(),
            destination.display()
        )
    })?;
    Ok(())
}

const fn worker_loader_script() -> &'static str {
    r#"import init from "./smarts_evolution_web_worker.js";

await init({ module_or_path: new URL("./smarts_evolution_web_worker_bg.wasm", import.meta.url) });
"#
}
