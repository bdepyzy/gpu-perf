import math

import torch

from benchmark import common
from benchmark.common import (
    PROBLEMS,
    bench_median,
    compare,
    format_percent,
    format_ratio,
    format_time,
    make_models,
    print_table,
    problem_modules,
    roofline_us,
    to_cuda,
)


DEPS = ("flashinfer-python[cu13]", "nvidia-cudnn-frontend")
SHAPES = [
    {
        "batch": 8,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "seq_len": 1024,
        "page_size": 16,
    },
    {
        "batch": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "seq_len": 2048,
        "page_size": 16,
    },
    {
        "batch": 4,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "seq_len": 4096,
        "page_size": 16,
    },
    {
        "batch": 16,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "seq_len": 1535,
        "page_size": 16,
    },
    {
        "batch": 8,
        "num_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 64,
        "seq_len": 2000,
        "page_size": 16,
    },
    {
        "batch": 4,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "seq_len": 32768,
        "page_size": 16,
    },
]


class PagedAttentionBaselines:
    """Prepared cuDNN and FlashInfer calls with setup excluded from timing."""

    def __init__(self, inputs, shape):
        from flashinfer.decode import (
            BatchDecodeWithPagedKVCacheWrapper,
            cudnn_batch_decode_with_kv_cache,
        )

        query, kv_cache, block_table, seq_lens = inputs
        batch = shape["batch"]
        heads = shape["num_heads"]
        kv_heads = shape["num_kv_heads"]
        head_dim = shape["head_dim"]
        page_size = shape["page_size"]
        scale = 1.0 / math.sqrt(head_dim)

        # Layout conversion is model-level KV-cache preparation, not attention
        # execution, so it is intentionally outside the timed calls.
        flash_k = kv_cache[..., :head_dim].contiguous()
        flash_v = kv_cache[..., head_dim:].contiguous()
        cudnn_k = flash_k.permute(0, 2, 1, 3).contiguous()
        cudnn_v = flash_v.permute(0, 2, 1, 3).contiguous()

        pages_per_seq = ((seq_lens + page_size - 1) // page_size).to(torch.int32)
        indptr = torch.zeros(batch + 1, dtype=torch.int32, device=query.device)
        indptr[1:] = torch.cumsum(pages_per_seq, dim=0)
        indices = torch.cat([block_table[b, : int(pages_per_seq[b].item())] for b in range(batch)]).to(torch.int32)
        last_page_len = ((seq_lens - 1) % page_size + 1).to(torch.int32)

        flash_workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=query.device)
        self.flashinfer = BatchDecodeWithPagedKVCacheWrapper(flash_workspace, kv_layout="NHD", backend="cute-dsl")
        self.flashinfer.plan(
            indptr,
            indices,
            last_page_len,
            num_qo_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            q_data_type=query.dtype,
            kv_data_type=kv_cache.dtype,
            o_data_type=query.dtype,
            sm_scale=scale,
        )
        self.flashinfer_name = f"FlashInfer/{self.flashinfer._backend}"
        self.flashinfer_out = torch.empty_like(query)
        self._flash_query = query
        self._flash_kv = (flash_k, flash_v)

        self._cudnn = cudnn_batch_decode_with_kv_cache
        self._cudnn_query = query
        self._cudnn_k = cudnn_k
        self._cudnn_v = cudnn_v
        self._cudnn_scale = scale
        self._cudnn_workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=query.device)
        self._cudnn_seq_lens = seq_lens.reshape(batch, 1, 1, 1)
        self._cudnn_block_table = block_table.contiguous()
        self.cudnn_out = torch.empty_like(query)
        self._max_sequence = shape["seq_len"]

        # Build/JIT both library kernels before correctness checks and timing.
        self.run_cudnn()
        self.run_flashinfer()
        torch.cuda.synchronize()

    def run_cudnn(self):
        return self._cudnn(
            self._cudnn_query,
            self._cudnn_k,
            self._cudnn_v,
            self._cudnn_scale,
            self._cudnn_workspace,
            max_sequence_kv=self._max_sequence,
            actual_seq_lens_kv=self._cudnn_seq_lens,
            block_tables=self._cudnn_block_table,
            out=self.cudnn_out,
        )

    def run_flashinfer(self):
        if self.flashinfer._backend == "cute-dsl":
            self.flashinfer_out.zero_()
        return self.flashinfer.run(self._flash_query, self._flash_kv, out=self.flashinfer_out)


def evaluate(workload=None):
    _, reference, solution = problem_modules("paged_attention")
    meta = PROBLEMS["paged_attention"]
    sol_ratios = []
    library_ratios = []
    rows = []
    flashinfer_name = None

    for index, shape in enumerate(SHAPES):
        reference_model, solution_model = make_models("paged_attention", reference, solution, shape)
        torch.manual_seed(2026)
        torch.cuda.manual_seed_all(2026)
        inputs = to_cuda(reference.get_inputs())
        with torch.no_grad():
            reference_out = reference_model(*inputs)
            solution_out = solution_model(*inputs)
        ok, message = compare(reference_out, solution_out, meta["tol"])
        if not ok:
            raise RuntimeError(f"solution correctness failed | shape={index} | {message}")

        baselines = PagedAttentionBaselines(inputs, shape)
        for name, output in (
            ("cuDNN", baselines.cudnn_out),
            (baselines.flashinfer_name, baselines.flashinfer_out),
        ):
            ok, message = compare(reference_out, output, meta["tol"])
            if not ok:
                raise RuntimeError(f"{name} correctness failed | shape={index} | {message}")

        compact = f"B{shape['batch']} H{shape['num_heads']}/{shape['num_kv_heads']} D{shape['head_dim']} L{shape['seq_len']} P{shape['page_size']}"

        if common.CHECK_ONLY:
            print(f"ok | {compact}", flush=True)
            continue

        prepare = getattr(solution_model, "prepare_for_bench", None)
        solution_call = prepare(inputs) if prepare else (lambda: solution_model(*inputs))
        with torch.no_grad():
            solution_us = bench_median(solution_call)
            cudnn_us = bench_median(baselines.run_cudnn)
            flashinfer_us = bench_median(baselines.run_flashinfer)
        flops = eval(meta["flops"], {"__builtins__": {}}, shape)
        moved = eval(meta["bytes"], {"__builtins__": {}}, shape)
        sol_us = roofline_us(flops, moved, meta["peak"])
        best_us = min(cudnn_us, flashinfer_us)
        sol_ratios.append(sol_us / solution_us)
        library_ratios.append(best_us / solution_us)
        flashinfer_name = baselines.flashinfer_name
        rows.append((compact, format_time(solution_us), format_time(sol_us), format_percent(100 * sol_us / solution_us), format_time(cudnn_us), format_time(flashinfer_us), format_ratio(library_ratios[-1])))

    if common.CHECK_ONLY:
        print("all shapes correct", flush=True)
        return

    sol_gmean = math.exp(sum(math.log(max(x, 1e-9)) for x in sol_ratios) / len(sol_ratios))
    library_gmean = math.exp(sum(math.log(max(x, 1e-9)) for x in library_ratios) / len(library_ratios))
    title = f"{torch.cuda.get_device_name()}  |  paged_attention"
    footer = ("Geomean", "", "", format_percent(100 * sol_gmean), "", "", format_ratio(library_gmean))
    print_table(title, "", ("Shape", "Kernel", "SOL", "SOL eff.", "cuDNN", flashinfer_name, "Perf. vs best"), rows, footer)
