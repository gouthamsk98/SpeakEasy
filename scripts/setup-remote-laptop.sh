#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
#
# One-shot setup for a speaker's Ubuntu laptop (22.04 / 24.04):
#   1. Tailscale (+ Tailscale SSH and OpenSSH) so the trainer can reach the laptop remotely
#   2. build tools, audio/GUI libraries, Rust
#   3. SpeakEasy: clone, ONNX Runtime (GPU if an NVIDIA card is present, else CPU),
#      the pretrained base model, a release build, and a desktop launcher
#
# Run it as the normal desktop user (NOT with sudo); it asks for sudo when needed:
#   curl -fsSL https://raw.githubusercontent.com/gouthamsk98/SpeakEasy/main/scripts/setup-remote-laptop.sh -o setup.sh
#   bash setup.sh                         # interactive Tailscale login (prints a URL)
#   bash setup.sh --authkey tskey-auth-…  # unattended login with a Tailscale auth key
#
# Options:
#   --authkey KEY     Tailscale auth key (or env TS_AUTHKEY). Without it, a login URL is printed.
#   --hostname NAME   Tailscale machine name (default: speakeasy-<user>)
#   --dir PATH        where to put the code (default: ~/SpeakEasy)
#   --repo URL        git repository (default: the public SpeakEasy repo)
#   --branch NAME     branch to check out (default: main)
#   --cpu             force the CPU build even if an NVIDIA GPU is present
#   --no-model        skip downloading the pretrained base model (~430 MB GPU / ~100 MB CPU)
#   --keep-awake      stop the laptop from suspending (so it stays reachable remotely)
#   --tailscale-only  only set up Tailscale + SSH, skip SpeakEasy
#   --skip-tailscale  only set up SpeakEasy
# Safe to re-run: every step checks what is already done.
set -euo pipefail

REPO="https://github.com/gouthamsk98/SpeakEasy.git"
BRANCH="main"
DIR="$HOME/SpeakEasy"
AUTHKEY="${TS_AUTHKEY:-}"
TS_HOST="speakeasy-$(id -un | tr -c 'a-zA-Z0-9-\n' '-')"
FORCE_CPU=0 NO_MODEL=0 KEEP_AWAKE=0 DO_TS=1 DO_APP=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --authkey) AUTHKEY="$2"; shift 2 ;;
    --hostname) TS_HOST="$2"; shift 2 ;;
    --dir) DIR="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --cpu) FORCE_CPU=1; shift ;;
    --no-model) NO_MODEL=1; shift ;;
    --keep-awake) KEEP_AWAKE=1; shift ;;
    --tailscale-only) DO_APP=0; shift ;;
    --skip-tailscale) DO_TS=0; shift ;;
    -h|--help) sed -n '3,27p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 1 ;;
  esac
done

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -ne 0 ]] || die "run as your normal user, not root/sudo (Rust and the app are installed per user)"
[[ -r /etc/os-release ]] && . /etc/os-release
[[ "${ID:-}" == ubuntu || "${ID_LIKE:-}" == *ubuntu* ]] || warn "this script is written for Ubuntu; continuing anyway (${PRETTY_NAME:-unknown OS})"
[[ "$(uname -m)" == x86_64 ]] || die "only x86_64 is supported (ONNX Runtime builds are x86_64)"

step "asking for sudo once (kept alive while the script runs)"
sudo -v
while true; do sudo -n true; sleep 50; kill -0 "$$" 2>/dev/null || exit; done 2>/dev/null &
SUDO_KEEPALIVE=$!
trap 'kill $SUDO_KEEPALIVE 2>/dev/null || true' EXIT

export DEBIAN_FRONTEND=noninteractive
step "installing system packages"
sudo apt-get update -y
sudo apt-get install -y \
  curl ca-certificates git openssh-server pciutils \
  build-essential pkg-config cmake \
  libasound2-dev libxkbcommon-dev libxkbcommon-x11-0 libwayland-dev libx11-dev \
  libxcursor1 libxrandr2 libxi6 libvulkan1 mesa-vulkan-drivers libgl1 \
  xdg-desktop-portal xdg-desktop-portal-gtk zenity pulseaudio-utils \
  python3 python3-pip python3-venv

# ---------------------------------------------------------------- remote access
if [[ $DO_TS -eq 1 ]]; then
  step "OpenSSH server (fallback when Tailscale SSH is not used)"
  sudo systemctl enable --now ssh

  step "Tailscale"
  if ! command -v tailscale >/dev/null; then
    curl -fsSL https://tailscale.com/install.sh | sh
  else
    echo "already installed: $(tailscale version | head -1)"
  fi
  sudo systemctl enable --now tailscaled

  if tailscale status >/dev/null 2>&1; then
    echo "already logged in; re-applying settings"
    sudo tailscale set --ssh --hostname="$TS_HOST"
  else
    UP=(sudo tailscale up --ssh --hostname="$TS_HOST")
    [[ -n "$AUTHKEY" ]] && UP+=(--authkey="$AUTHKEY")
    [[ -z "$AUTHKEY" ]] && echo ">>> open the URL printed below in a browser and log in to YOUR Tailscale account <<<"
    "${UP[@]}"
  fi
  # let the desktop user run `tailscale` without sudo
  sudo tailscale set --operator="$(id -un)" 2>/dev/null || true

  if [[ $KEEP_AWAKE -eq 1 ]]; then
    step "disabling suspend so the laptop stays reachable"
    sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
    sudo mkdir -p /etc/systemd/logind.conf.d
    printf '[Login]\nHandleLidSwitch=ignore\nHandleLidSwitchExternalPower=ignore\nHandleLidSwitchDocked=ignore\n' \
      | sudo tee /etc/systemd/logind.conf.d/50-speakeasy-keep-awake.conf >/dev/null
    echo "lid close no longer suspends (takes effect after reboot). Undo: sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target"
  fi
fi

# ---------------------------------------------------------------- SpeakEasy
REBOOT_NEEDED=0
if [[ $DO_APP -eq 1 ]]; then
  step "GPU check"
  GPU=cpu
  if [[ $FORCE_CPU -eq 0 ]] && lspci | grep -qi 'vga.*nvidia\|3d.*nvidia'; then
    GPU=cuda
    if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
      echo "NVIDIA GPU found but no working driver; installing the recommended one"
      sudo apt-get install -y ubuntu-drivers-common
      sudo ubuntu-drivers install
      REBOOT_NEEDED=1
    fi
  fi
  echo "inference device: $GPU"

  step "Rust toolchain"
  if [[ ! -x "$HOME/.cargo/bin/cargo" ]]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  fi
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
  rustup update stable --no-self-update
  cargo --version

  step "SpeakEasy source -> $DIR"
  if [[ -d "$DIR/.git" ]]; then
    git -C "$DIR" fetch origin
    git -C "$DIR" checkout "$BRANCH"
    git -C "$DIR" pull --ff-only origin "$BRANCH" || warn "could not fast-forward (local changes?); building what is there"
  else
    git clone --branch "$BRANCH" "$REPO" "$DIR"
  fi
  cd "$DIR"

  step "ONNX Runtime ($GPU)"
  if [[ $GPU == cuda ]]; then
    [[ -f third_party/onnxruntime/lib/libonnxruntime_providers_cuda.so && -f third_party/cudnn/lib/libcudnn.so.9 ]] \
      && echo "already present" || bash scripts/fetch-onnxruntime.sh gpu
  else
    [[ -f third_party/onnxruntime/lib/libonnxruntime.so ]] && echo "already present" || bash scripts/fetch-onnxruntime.sh cpu
  fi

  if [[ $GPU == cuda ]]; then
    # The ORT CUDA provider links against the CUDA 12 runtime + cuBLAS + cuRAND. The NVIDIA
    # driver does not ship these; rather than the multi-GB CUDA toolkit, take them from
    # NVIDIA's PyPI wheels (~600 MB). The launcher puts this folder on LD_LIBRARY_PATH.
    step "CUDA 12 runtime libraries"
    if ldconfig -p | grep -q 'libcublasLt.so.12' && ldconfig -p | grep -q 'libcudart.so.12'; then
      echo "system CUDA 12 libraries found"
    elif [[ -f third_party/cuda/lib/libcublasLt.so.12 ]]; then
      echo "already present"
    else
      TMPW="$(mktemp -d)"
      python3 -m pip download --quiet --no-deps --only-binary=:all: --platform manylinux_2_27_x86_64 \
        --python-version 3.10 -d "$TMPW" \
        "nvidia-cuda-runtime-cu12>=12,<13" "nvidia-cublas-cu12>=12,<13" "nvidia-curand-cu12>=10,<11"
      mkdir -p third_party/cuda/lib
      for w in "$TMPW"/*.whl; do python3 -m zipfile -e "$w" "$TMPW/x"; done
      find "$TMPW/x/nvidia" -path '*/lib/*.so*' -exec cp {} third_party/cuda/lib/ \;
      rm -rf "$TMPW"
      echo "installed -> third_party/cuda/lib"
    fi
  fi

  if [[ $NO_MODEL -eq 0 ]]; then
    step "pretrained base model (used until a personal model is imported)"
    if ls models/*/model.json >/dev/null 2>&1; then
      echo "a model is already installed: $(ls -d models/*/ | head -3 | tr '\n' ' ')"
    else
      # the importer only needs the `onnx` package; keep it in the research venv
      [[ -x research/.venv/bin/python ]] || python3 -m venv research/.venv
      research/.venv/bin/python -m pip install --quiet --upgrade pip onnx
      if [[ $GPU == cuda ]]; then bash scripts/fetch-demo-model.sh; else bash scripts/fetch-demo-model.sh int8; fi
    fi
  fi

  step "building the app (release; first build takes several minutes)"
  FEAT=(); [[ $GPU == cuda ]] && FEAT=(--features cuda)
  # cap parallel jobs on small laptops: the LTO link is memory hungry
  JOBS=$(( $(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo) / 3 )); (( JOBS < 1 )) && JOBS=1
  (( JOBS > $(nproc) )) && JOBS=$(nproc)
  cargo build --release -j "$JOBS" -p asr-app -p asr-cli "${FEAT[@]}"

  step "launcher"
  mkdir -p "$HOME/.local/bin" "$HOME/.local/share/applications"
  cat > "$HOME/.local/bin/speak-easy" <<EOF
#!/usr/bin/env bash
# SpeakEasy launcher (written by setup-remote-laptop.sh)
[[ -d "$DIR/third_party/cuda/lib" ]] && export LD_LIBRARY_PATH="$DIR/third_party/cuda/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
exec "$DIR/target/release/speak-easy" "\${1:-auto}" "\${2:-auto}"
EOF
  chmod +x "$HOME/.local/bin/speak-easy"
  cat > "$HOME/.local/bin/asr" <<EOF
#!/usr/bin/env bash
# SpeakEasy command-line tools (written by setup-remote-laptop.sh)
[[ -d "$DIR/third_party/cuda/lib" ]] && export LD_LIBRARY_PATH="$DIR/third_party/cuda/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
exec "$DIR/target/release/asr" "\$@"
EOF
  chmod +x "$HOME/.local/bin/asr"
  cat > "$HOME/.local/share/applications/speakeasy.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=SpeakEasy
Comment=Live captions trained on your own voice
Exec=$HOME/.local/bin/speak-easy
Terminal=false
Categories=Utility;Accessibility;AudioVideo;
EOF
  if [[ -d "$HOME/Desktop" ]]; then
    cp "$HOME/.local/share/applications/speakeasy.desktop" "$HOME/Desktop/"
    chmod +x "$HOME/Desktop/speakeasy.desktop"
    gio set "$HOME/Desktop/speakeasy.desktop" metadata::trusted true 2>/dev/null || true
  fi

  step "self-check"
  "$HOME/.local/bin/asr" devices || warn "no audio input devices listed (plug in / unmute the mic)"
  "$HOME/.local/bin/asr" models || true
fi

# ---------------------------------------------------------------- summary
step "done"
if [[ $DO_TS -eq 1 ]]; then
  TS_IP="$(tailscale ip -4 2>/dev/null | head -1 || true)"
  echo "Tailscale name : $TS_HOST"
  echo "Tailscale IP   : ${TS_IP:-not connected yet}"
  echo "Remote login   : ssh $(id -un)@$TS_HOST      (or ssh $(id -un)@${TS_IP:-<ip>})"
  echo "Copy files     : scp speakeasy-model-*.zip $(id -un)@$TS_HOST:~/Downloads/"
  echo "               : scp $(id -un)@$TS_HOST:'~/Downloads/speakeasy-dataset-*.zip' ."
fi
if [[ $DO_APP -eq 1 ]]; then
  echo "Run the app    : SpeakEasy in the app menu / desktop, or 'speak-easy' in a terminal"
  echo "Update later   : bash $DIR/scripts/setup-remote-laptop.sh --skip-tailscale"
fi
[[ $REBOOT_NEEDED -eq 1 ]] && printf '\n\033[1;33mREBOOT NOW\033[0m to load the NVIDIA driver (sudo reboot). The app will use the GPU after that.\n'
exit 0
