#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
container_name=${CONTAINER_NAME:-recsys-example-ubuntu22}
workspace=/workspace/recsys-npu-wrapper

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not installed or is not on PATH" >&2
    exit 1
fi

device_args=()
for device in /dev/davinci[0-9]*; do
    if [[ $device =~ ^/dev/davinci[0-9]+$ && -c $device ]]; then
        device_args+=(--device "$device")
    fi
done
if ((${#device_args[@]} == 0)); then
    echo "No Ascend NPU device found at /dev/davinciN" >&2
    exit 1
fi

for device in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
    if [[ ! -c $device ]]; then
        echo "Required Ascend device not found: $device" >&2
        exit 1
    fi
    device_args+=(--device "$device")
done

for path in /usr/local/Ascend /usr/local/sbin /usr/local/bin; do
    if [[ ! -d $path ]]; then
        echo "Required host directory not found: $path" >&2
        exit 1
    fi
done
for path in /etc/hccn.conf /etc/ascend_install.info; do
    if [[ ! -f $path ]]; then
        echo "Required host file not found: $path" >&2
        exit 1
    fi
done

mount_args=(
    -v "/usr/local/Ascend:/usr/local/Ascend"
    -v "/etc/hccn.conf:/etc/hccn.conf"
    -v "/etc/ascend_install.info:/etc/ascend_install.info"
    -v "/usr/local/sbin:/usr/local/sbin"
    -v "/usr/local/bin:/usr/local/bin"
)

docker run -itd --net=host --privileged \
    --name "$container_name" \
    "${device_args[@]}" \
    "${mount_args[@]}" \
    --mount "type=bind,source=$repo_root,target=$workspace" \
    --workdir "$workspace" \
    ubuntu:22.04 sleep infinity

echo "Container created: $container_name"
echo "Enter it with: docker exec -it $container_name bash"
