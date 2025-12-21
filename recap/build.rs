use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get git commit
    let git_commit = Command::new("git")
        .args(["describe", "--always"])
        .output()?;

    if !git_commit.status.success() {
        return Err("`git describe --always` command failed".into());
    }

    let git_commit = String::from_utf8(git_commit.stdout)?.trim().to_string();
    println!("cargo:rustc-env=GIT_COMMIT={git_commit}");

    // Read Cargo.toml to extract version information
    let cargo_toml_path =
        std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR")?).join("Cargo.toml");
    let cargo_toml_content = std::fs::read_to_string(cargo_toml_path)?;

    // Parse the TOML content
    let cargo_toml: toml::Value = toml::from_str(&cargo_toml_content)?;

    // Extract custom versions from metadata and set environment variables
    if let Some(metadata) = cargo_toml["package"]["metadata"]["versions"].as_table() {
        if let Some(recap_version) = metadata["recap_version"].as_integer() {
            println!("cargo:rustc-env=RECAP_VERSION={recap_version}");
        }
        if let Some(proto_version) = metadata["proto_version"].as_integer() {
            println!("cargo:rustc-env=PROTO_VERSION={proto_version}");
        }
    }

    Ok(())
}
