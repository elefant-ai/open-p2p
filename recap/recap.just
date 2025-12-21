export ELEFANT_TRUNK_ROOT := `git rev-parse --show-toplevel`
export PATH := env("PATH") + ";" + `git rev-parse --show-toplevel` + "/.cargo-third-party/gstreamer/bin"

# list all available recipes
help:
    @just --list recap

# setup the workspace by installing the gstreamer libraries
@_setup_workspace:
    cargo run -p gstreamer-deps --bin setup_workspace --features="build-binary"

# run in release mode for better performance
run: _setup_workspace
    cargo run -p recap --release --bin recap

# run in debug mode with with recap logs at trace level
trace:
    GST_DEBUG="3" RUST_LOG="warn,iced_wgpu::window::compositor=info,recap=trace" cargo run -p recap --release --bin recap --features trace

# run in dev mode for faster compilation
dev: _setup_workspace
    cargo run -p recap --bin recap

# run encoding check - all captures will automatically be stopped after 60s
encoding-check: _setup_workspace
    cargo run -p recap --release --bin recap --features encoding-check


convert args="":
    cargo run -p recap --bin convert --features convert -- {{args}}

watch:
    cargo watch -x 'run'


build-release:
    #!/usr/bin/env nu
    let version = open Cargo.toml | get package.version

    # Fetch signing secrets and set Azure environment variables
    let secrets = (just setec_get /prod/client/windows_signing_secrets | complete)
    if $secrets.exit_code != 0 {
        error make { msg: "Failed to fetch signing secrets" }
    }

    # Parse the export statements and set Azure environment variables
    let azure_vars = ($secrets.stdout
        | lines
        | where ($it | str starts-with "export AZURE")
        | each { |line|
            let parts = ($line | str replace "export " "" | split column "=" name value)
            let env_name = ($parts.name.0 | str trim)
            let env_value = ($parts.value.0 | str trim -c '"')
            {name: $env_name, value: $env_value}
        })

    # Set each Azure environment variable
    for var in $azure_vars {
        load-env {($var.name): ($var.value)}
    }

    cargo build -p recap -r --no-default-features
    cp ../../../target/release/recap.exe ./recap.exe
    let recap = realpath ./recap.exe
    just client sign $"($recap)"
    cp ./recap.exe $"./recap-($version).exe"
