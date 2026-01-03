#!/bin/bash
set -e

REPO="alDuncanson/latent"
INSTALL_DIR="/usr/local/bin"

detect_platform() {
    local os arch

    case "$(uname -s)" in
        Linux*)  os="linux" ;;
        Darwin*) os="darwin" ;;
        MINGW*|MSYS*|CYGWIN*) os="windows" ;;
        *) echo "Unsupported OS: $(uname -s)" >&2; exit 1 ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64) arch="amd64" ;;
        arm64|aarch64) arch="arm64" ;;
        *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
    esac

    echo "${os}-${arch}"
}

get_latest_version() {
    curl -sS "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name"' | cut -d'"' -f4
}

main() {
    local platform version binary_name download_url tmp_dir

    platform=$(detect_platform)
    version=$(get_latest_version)

    if [ -z "$version" ]; then
        echo "Error: Could not determine latest version" >&2
        exit 1
    fi

    binary_name="latent-${platform}"
    download_url="https://github.com/${REPO}/releases/download/${version}/${binary_name}"

    echo "Installing latent ${version} for ${platform}..."

    tmp_dir=$(mktemp -d)
    trap 'rm -rf "$tmp_dir"' EXIT

    if ! curl -sSL -o "${tmp_dir}/latent" "$download_url"; then
        echo "Error: Failed to download ${download_url}" >&2
        exit 1
    fi

    chmod +x "${tmp_dir}/latent"

    if [ -w "$INSTALL_DIR" ]; then
        mv "${tmp_dir}/latent" "${INSTALL_DIR}/latent"
    else
        echo "Installing to ${INSTALL_DIR} (requires sudo)..."
        sudo mv "${tmp_dir}/latent" "${INSTALL_DIR}/latent"
    fi

    echo "Installed latent ${version} to ${INSTALL_DIR}/latent"
    echo "Run 'latent' to start"
}

main
