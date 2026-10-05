import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from paged_attention.reference import Model, get_inputs
from utils import benchmark_utils as bench

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

app = bench.create_app(__file__, ("flashinfer-python[cu13]", "nvidia-cudnn-frontend"), nvcc=True)


def prepare_baselines(inputs, shape):
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper, cudnn_batch_decode_with_kv_cache

    query, kv_cache, block_table, seq_lens = inputs
    batch, heads, kv_heads = shape["batch"], shape["num_heads"], shape["num_kv_heads"]
    head_dim, page_size = shape["head_dim"], shape["page_size"]
    scale = head_dim ** -0.5

    flash_k = kv_cache[..., :head_dim].contiguous()
    flash_v = kv_cache[..., head_dim:].contiguous()
    cudnn_k = flash_k.permute(0, 2, 1, 3).contiguous()
    cudnn_v = flash_v.permute(0, 2, 1, 3).contiguous()
    pages_per_seq = ((seq_lens + page_size - 1) // page_size).to(torch.int32)
    indptr = torch.zeros(batch + 1, dtype=torch.int32, device=query.device)
    indptr[1:] = torch.cumsum(pages_per_seq, dim=0)
    indices = torch.cat([block_table[b, :int(pages_per_seq[b].item())] for b in range(batch)]).to(torch.int32)
    last_page_len = ((seq_lens - 1) % page_size + 1).to(torch.int32)

    flash_calls = {}
    for backend in ("cute-dsl", "auto"):
        workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=query.device)
        wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD", backend=backend)
        wrapper.plan(
            indptr, indices, last_page_len, num_qo_heads=heads, num_kv_heads=kv_heads,
            head_dim=head_dim, page_size=page_size, q_data_type=query.dtype,
            kv_data_type=kv_cache.dtype, o_data_type=query.dtype, sm_scale=scale,
        )
        output = torch.empty_like(query)

        def decode(wrapper=wrapper, output=output):
            if wrapper._backend == "cute-dsl":
                output.zero_()
            return wrapper.run(query, (flash_k, flash_v), out=output)

        flash_calls[f"FlashInfer/{backend}"] = decode
    cudnn_workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=query.device)
    cudnn_seq_lens = seq_lens.reshape(batch, 1, 1, 1)
    cudnn_block_table = block_table.contiguous()
    cudnn_out = torch.empty_like(query)

    def run_cudnn():
        return cudnn_batch_decode_with_kv_cache(
            query, cudnn_k, cudnn_v, scale, cudnn_workspace,
            max_sequence_kv=shape["seq_len"], actual_seq_lens_kv=cudnn_seq_lens,
            block_tables=cudnn_block_table, out=cudnn_out,
        )

    return {"cuDNN": run_cudnn, **flash_calls}


@app.function(gpu=bench.B200_GPU, timeout=bench.B200_TIMEOUT)
@torch.no_grad()
def run(solution_file: str, check: bool = False, shape: int | None = None):
    solution = bench.load("paged_attention", solution_file)
    sol_ratios = []
    library_ratios = []
    rows = []

    for index, shape in enumerate(SHAPES):
        reference_model = Model(**shape).cuda().eval()
        torch.manual_seed(2026)
        inputs = [x.cuda() for x in get_inputs(**shape)]
        solution_out = torch.empty_like(inputs[0])
        tensors = (*inputs, solution_out)
        if hasattr(solution, "workspace_shape"):
            workspace_shape = solution.workspace_shape(shape["batch"], shape["num_heads"],
                                                       shape["head_dim"], shape["seq_len"])
            tensors += (torch.empty(workspace_shape, dtype=torch.float32, device="cuda"),)
        args = tuple(from_dlpack(t, assumed_align=16) for t in tensors)
        compiled = cute.compile(solution.paged_attention, *args)
        reference_out = reference_model(*inputs)
        compiled(*args)
        ok, message = bench.compare(reference_out, solution_out, 0.02)
        if not ok:
            raise RuntimeError(f"solution correctness failed | shape={index} | {message}")

        baselines = prepare_baselines(inputs, shape)
        for name, fn in baselines.items():
            ok, message = bench.compare(reference_out, fn(), 0.02)
            if not ok:
                raise RuntimeError(f"{name} correctness failed | shape={index} | {message}")

        compact = f"B{shape['batch']} H{shape['num_heads']}/{shape['num_kv_heads']} D{shape['head_dim']} L{shape['seq_len']} P{shape['page_size']}"

        if check:
            print(f"ok | {compact}", flush=True)
            continue

        solution_us = bench.bench_median(lambda: compiled(*args))
        times = {name: bench.bench_median(fn) for name, fn in baselines.items()}
        flops = 4*shape["batch"]*shape["num_heads"]*shape["seq_len"]*shape["head_dim"]
        moved = 4*shape["batch"]*shape["head_dim"]*(shape["seq_len"]*shape["num_kv_heads"] + shape["num_heads"])
        sol_us = bench.roofline_us(flops, moved, "bf16")
        best = min(times, key=times.get)
        best_us = times[best]
        sol_ratios.append(sol_us / solution_us)
        library_ratios.append(best_us / solution_us)
        rows.append((compact, bench.format_time(solution_us), bench.format_time(sol_us),
                     bench.format_percent(100 * sol_us / solution_us),
                     *(bench.format_time(t) for t in times.values()), best,
                     bench.format_ratio(library_ratios[-1])))

    if check:
        print("all shapes correct", flush=True)
        return

    sol_gmean = bench.geomean(sol_ratios)
    library_gmean = bench.geomean(library_ratios)
    title = f"{torch.cuda.get_device_name()}  |  paged_attention"
    footer = ("Geomean", "", "", bench.format_percent(100 * sol_gmean), "", "", "", "", bench.format_ratio(library_gmean))
    bench.print_table(title, "Speedup against the fastest measured library per shape",
                      ("Shape", "Kernel", "Roofline", "Roof eff.", "cuDNN", "FlashInfer/cute", "FlashInfer/auto", "Best", "Speedup"), rows, footer)


@app.local_entrypoint()
def main(*args):
    bench.launch(run, __file__, args, supports_shape=False)
