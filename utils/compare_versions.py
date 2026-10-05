from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parent.parent
image = modal.Image.debian_slim(python_version='3.13').env({'PYTHONPATH': '/workspace'}).uv_pip_install(
    'torch==2.11.0', 'nvidia-cutlass-dsl[cu13]==4.7.0', 'numpy==2.5.2', 'einops==0.8.2')
for name in ('utils', 'fp6_gemm', 'kda', 'paged_attention', 'topk_bitonic'):
    image = image.add_local_dir(ROOT / name, '/workspace/' + name)
app = modal.App('compare-kernel-versions', image=image)

@app.function(gpu='B200', cpu=4, timeout=1800)
def compare(suite):
    import hashlib
    import importlib
    import statistics
    import torch
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack
    from utils import benchmark_utils as bench
    benchmark = importlib.import_module(suite + '.benchmark')
    reference = importlib.import_module(suite + '.reference')
    versions = [importlib.import_module(suite + '.' + v) for v in ('v1', 'v2')]
    rows = []
    torch.set_grad_enabled(False)
    for dims in benchmark.SHAPES:
        torch.manual_seed(2026)
        model = None
        if suite == 'fp6_gemm':
            model = reference.Model(**dims).cuda()
            x = reference.get_inputs(dims['M'], dims['K'])[0].cuda()
            inputs = (x, model.w_q)
            expected = model(x)
        else:
            inputs = tuple(t.cuda() for t in reference.get_inputs(**dims))
            if suite == 'topk_bitonic':
                expected = torch.topk(inputs[0], dims['k'])
            else:
                model = (reference.Model(*(dims[key] for key in ('B','T','H','K','V','CHUNK_SIZE'))) if suite == 'kda' else reference.Model(**dims)).cuda()
                expected = model(*inputs)
        fns = []
        for solution in versions:
            if suite == 'topk_bitonic':
                values = torch.empty(dims['batch'], dims['k'], device='cuda')
                indices = torch.empty_like(values, dtype=torch.int64)
                shape = solution.workspace_shape(**dims)
                tensors = (*inputs, values, indices, torch.empty(shape, device='cuda'), torch.empty(shape,device='cuda',dtype=torch.int32))
                fn = solution.topk
                output = (values, indices)
            else:
                output = torch.empty_like(expected)
                tensors = (*inputs, output)
                fn = getattr(solution, {'fp6_gemm':'fp6','kda':'kda','paged_attention':'paged_attention'}[suite])
                if suite == 'paged_attention':
                    shape = solution.workspace_shape(dims['batch'], dims['num_heads'], dims['head_dim'], dims['seq_len'])
                    tensors += (torch.empty(shape,device='cuda'),)
            args = tuple(from_dlpack(t, assumed_align=16) for t in tensors)
            compiled = cute.compile(fn, *args)
            compiled(*args)
            torch.cuda.synchronize()
            if suite == 'topk_bitonic':
                ok, message = bench.compare_topk(inputs, expected, output, dims, 1e-4)
            else:
                ok, message = bench.compare(expected, output, .02 if suite == 'paged_attention' else .05)
            assert ok, (solution.__name__, dims, message)
            fns.append(lambda compiled=compiled,args=args,tensors=tensors: compiled(*args))
        times = [[], []]
        for order in ((0,1),(1,0)):
            for i in order:
                times[i].append(bench.bench_median(fns[i]))
        medians = [statistics.median(t) for t in times]
        row = dict(shape=dims, v1_us=medians[0], v2_us=medians[1], speedup=medians[0]/medians[1], correct=True)
        rows.append(row)
        print(json.dumps(row), flush=True)
    return dict(gpu=torch.cuda.get_device_name(), method='CUDA events; 256 MiB L2 flush; 10 warmups, 30 iterations; median of AB/BA runs',
                source_sha256={v.__name__:hashlib.sha256(Path(v.__file__).read_bytes()).hexdigest() for v in versions},
                geomean_speedup=bench.geomean([r['speedup'] for r in rows]), results=rows)

@app.local_entrypoint()
def main(suite: str):
    if suite not in ('fp6_gemm', 'kda', 'paged_attention', 'topk_bitonic'):
        raise ValueError(suite)
    result = compare.remote(suite)
    destination = ROOT / suite / 'v2_results.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print('Geomean speedup:', result['geomean_speedup'])
