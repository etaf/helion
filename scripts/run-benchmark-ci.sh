#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f pyproject.toml ]]; then
    echo "error: expected to run inside the helion repo" >&2
    exit 1
fi

DEFAULT_KERNELS="flash_attention,gdn_fwd_h,cross_entropy,grouped_gemm,gemm,welford,layer_norm-bwd,int4_gemm,softmax,layer_norm,mamba2_chunk_state,rms_norm-bwd,rope,jsd,mamba2_chunk_scan,kl_div,rms_norm"
SUITE="tritonbench"
BACKEND="${HELION_BACKEND:-triton}"
KERNELS="$DEFAULT_KERNELS"
ENV_VARS=""
CUSTOM_ARGS=""
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/test/test-reports}"

usage() {
    cat <<'EOF'
Usage: scripts/run-benchmark-ci.sh [options]

Mirror the TritonBench path from .github/workflows/benchmark.yml for local or
manual runs. By default, this runs the same GPU kernel list as
.github/workflows/benchmark_dispatch.yml.

Options:
  --suite <tritonbench|linattn>   Benchmark suite to run. Default: tritonbench
  --backend <name>                Helion backend. Default: HELION_BACKEND or triton
  --kernels <csv>                 Comma-separated kernel list
  --env-vars <string>             Extra env var assignments for the benchmark command
  --custom-args <string>          Extra arguments appended to the benchmark command
  --output-dir <path>             Report directory. Default: test/test-reports
  -h, --help                      Show this help

Examples:
  scripts/run-benchmark-ci.sh
  scripts/run-benchmark-ci.sh --kernels gemm,softmax
  scripts/run-benchmark-ci.sh --backend cute --kernels vector_add,layer_norm
  scripts/run-benchmark-ci.sh --env-vars 'HELION_AUTOTUNE_EFFORT=none' --custom-args '--num-inputs 2'
  scripts/run-benchmark-ci.sh --suite linattn --kernels vanilla_linear_attn,full_gla
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite)
            SUITE="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --kernels)
            KERNELS="$2"
            shift 2
            ;;
        --env-vars)
            ENV_VARS="$2"
            shift 2
            ;;
        --custom-args)
            CUSTOM_ARGS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "$SUITE" != "tritonbench" && "$SUITE" != "linattn" ]]; then
    echo "error: unsupported suite '$SUITE'" >&2
    exit 1
fi

export HELION_AUTOTUNE_LOG_LEVEL="${HELION_AUTOTUNE_LOG_LEVEL:-INFO}"
export HELION_AUTOTUNE_BUDGET_SECONDS="${HELION_AUTOTUNE_BUDGET_SECONDS:-5400}"
export HELION_BACKEND="$BACKEND"

mkdir -p "$OUTPUT_DIR"
rm -rf /tmp/torchinductor_*/ || true

run_tritonbench() {
    local kernel kernel_info impls baseline

    for kernel in ${KERNELS//,/ }; do
        echo "=========================================="
        echo "Running benchmark for kernel: $kernel"
        echo "=========================================="

        kernel_info="$(python benchmarks/run.py --device xpu --list-impls-for-benchmark-ci --op "$kernel" | grep "^$kernel:" || true)"
        impls="$(printf '%s\n' "$kernel_info" | perl -ne 'print "$1\n" if /impls=([^ ]*)/')"
        baseline="$(printf '%s\n' "$kernel_info" | perl -ne 'print "$1\n" if /baseline=([^ ]*)/')"

        if [[ -z "$impls" ]]; then
            echo "Warning: No implementations found for kernel $kernel, skipping..."
            continue
        fi
        if [[ -z "$baseline" ]]; then
            echo "Warning: No baseline found for kernel $kernel, skipping..."
            continue
        fi

        echo "Using baseline: $baseline"
        echo "Available implementations for $kernel: $impls"

        env \
            ${ENV_VARS:+$ENV_VARS }\
            HELION_PRINT_OUTPUT_CODE=1 \
            HELION_AUTOTUNE_LOG="$OUTPUT_DIR/autotune-$kernel" \
            python benchmarks/run.py \
                --device xpu  \
                --op "$kernel" \
                --helion-backend "$BACKEND" \
                --metrics speedup,accuracy,latency \
                --measure-compile-time \
                --latency-measure-mode triton_do_bench \
                --only "$impls" \
                --only-match-mode prefix-with-baseline \
                --baseline "$baseline" \
                --atol 1e-2 \
                --rtol 1e-2 \
                --input-sample-mode equally-spaced-k \
                --output "$OUTPUT_DIR/helionbench.json" \
                --append-to-output \
                --autotune-metrics-json "$OUTPUT_DIR/autotune-metrics-$kernel.json" \
                --keep-going \
                --num-inputs 1
                ${CUSTOM_ARGS}

        echo "✅ Completed benchmark for kernel: $kernel"
    done
}

run_linattn() {
    local -a kernel_flag=()

    if [[ "$KERNELS" != "all" && -n "$KERNELS" ]]; then
        kernel_flag=(--kernel "$KERNELS")
    fi

    echo "=========================================="
    echo "Linear-Attention Benchmark"
    echo "Kernels: ${KERNELS:-all}"
    echo "=========================================="

    env \
        ${ENV_VARS:+$ENV_VARS }\
        HELION_PRINT_OUTPUT_CODE=1 \
        python -m benchmarks.run_linattn \
            "${kernel_flag[@]}" \
            --output "$OUTPUT_DIR/helionbench.json" \
            ${CUSTOM_ARGS}
}

case "$SUITE" in
    tritonbench)
        run_tritonbench
        ;;
    linattn)
        run_linattn
        ;;
esac

if [[ ! -s "$OUTPUT_DIR/helionbench.json" ]]; then
    echo "❌ helionbench.json is missing or empty" >&2
    exit 1
fi

cat "$OUTPUT_DIR/helionbench.json"
