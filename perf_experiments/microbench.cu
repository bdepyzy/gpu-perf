#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cooperative_groups.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) \
    throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e)); } while (0)

using Clock = std::chrono::steady_clock;
using U64 = unsigned long long;
namespace cg = cooperative_groups;

struct Result { U64 cycles; unsigned value; int status; };

void row(const std::string& suite, const std::string& name, int sample,
         const std::string& metric, double value, const char* unit,
         int iterations = 1, size_t bytes = 0, int stride = 0, int threads = 1, int nodes = 1) {
    std::cout << suite << ',' << name << ',' << sample << ',' << metric << ','
              << std::setprecision(12) << value << ',' << unit << ',' << iterations << ','
              << bytes << ',' << stride << ',' << threads << ',' << nodes << '\n';
}

double micros(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

__global__ void empty_kernel() { asm volatile(""); }
__global__ void tiny_kernel(unsigned* out) { out[threadIdx.x] += 1; }
__global__ void cdp_launcher(int count, Result* out) {
    U64 start = clock64();
    for (int i = 0; i < count; ++i) empty_kernel<<<1, 1>>>();
    U64 end = clock64();
    out->cycles = end - start;
    out->status = int(cudaGetLastError());
}
__global__ void graph_launcher(cudaGraphExec_t child, Result* out) {
    U64 start = clock64();
    cudaError_t status = cudaGraphLaunch(child, cudaStreamGraphTailLaunch);
    U64 end = clock64();
    out->cycles = end - start;
    out->status = int(status);
}

struct Events {
    cudaEvent_t begin, end;
    Events() { CUDA(cudaEventCreate(&begin)); CUDA(cudaEventCreate(&end)); }
    ~Events() { cudaEventDestroy(begin); cudaEventDestroy(end); }
};

struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    Graph(cudaStream_t stream, int nodes, bool tiny, unsigned* out, bool device = false) {
        CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int n = 0; n < nodes; ++n) {
            if (tiny) tiny_kernel<<<1, 32, 0, stream>>>(out);
            else empty_kernel<<<1, 1, 0, stream>>>();
        }
        CUDA(cudaStreamEndCapture(stream, &graph));
        CUDA(cudaGraphInstantiateWithFlags(&exec, graph, device ? cudaGraphInstantiateFlagDeviceLaunch : 0));
        CUDA(cudaGraphUpload(exec, stream));
        CUDA(cudaStreamSynchronize(stream));
    }
    ~Graph() { cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); }
};

template<class Submit>
void launch_case(const std::string& name, Submit submit, cudaStream_t stream,
                 int samples, int repetitions, int nodes, int threads,
                 Result* device_result = nullptr) {
    Events events;
    for (int i = 0; i < 5; ++i) submit();
    CUDA(cudaGetLastError());
    CUDA(cudaStreamSynchronize(stream));
    for (int sample = 0; sample < samples; ++sample) {
        CUDA(cudaEventRecord(events.begin, stream));
        auto start = Clock::now();
        for (int i = 0; i < repetitions; ++i) submit();
        auto submitted = Clock::now();
        CUDA(cudaEventRecord(events.end, stream));
        CUDA(cudaEventSynchronize(events.end));
        CUDA(cudaGetLastError());
        float ms;
        CUDA(cudaEventElapsedTime(&ms, events.begin, events.end));
        row("launch", name, sample, "host_submit_us_per_call", micros(start, submitted) / repetitions,
            "us", repetitions, 0, 0, threads, nodes);
        row("launch", name, sample, "gpu_event_span_us_per_call", ms * 1000.0 / repetitions,
            "us", repetitions, 0, 0, threads, nodes);
        start = Clock::now();
        for (int i = 0; i < repetitions; ++i) submit();
        CUDA(cudaStreamSynchronize(stream));
        auto complete = Clock::now();
        row("launch", name, sample, "wall_completion_us_per_call", micros(start, complete) / repetitions,
            "us", repetitions, 0, 0, threads, nodes);
        if (device_result) {
            Result result;
            CUDA(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost));
            CUDA(cudaError_t(result.status));
            row("launch", name, sample, "device_submit_cycles_total", result.cycles,
                "cycles", 1, 0, 0, threads, nodes);
        }
    }
    std::cerr << "launch: " << name << " complete\n";
}

void launch_suite(int samples, bool quick, cudaStream_t stream) {
    unsigned* out;
    Result* result;
    CUDA(cudaMalloc(&out, 32 * sizeof(unsigned)));
    CUDA(cudaMalloc(&result, sizeof(Result)));
    CUDA(cudaMemset(out, 0, 32 * sizeof(unsigned)));
    for (bool tiny : {false, true}) {
        auto submit = [&] {
            if (tiny) tiny_kernel<<<1, 32, 0, stream>>>(out);
            else empty_kernel<<<1, 1, 0, stream>>>();
        };
        std::string base = tiny ? "host_tiny" : "host_empty";
        launch_case(base + "_single", submit, stream, samples, 1, 1, tiny ? 32 : 1);
        launch_case(base + "_batch", submit, stream, samples, quick ? 20 : 200, 1, tiny ? 32 : 1);
        for (int nodes : {1, 10, 100}) {
            Graph graph(stream, nodes, tiny, out);
            launch_case(std::string("host_graph_") + (tiny ? "tiny_" : "empty_") + std::to_string(nodes),
                        [&] { CUDA(cudaGraphLaunch(graph.exec, stream)); }, stream, samples,
                        quick ? 10 : 100, nodes, tiny ? 32 : 1);
        }
    }
    {
        Graph graph(stream, 0, false, out);
        launch_case("host_graph_zero_nodes", [&] { CUDA(cudaGraphLaunch(graph.exec, stream)); },
                    stream, samples, quick ? 10 : 100, 0, 0);
    }
    for (int nodes : {1, 10, 100}) {
        launch_case("device_cdp_" + std::to_string(nodes),
                    [&] { cdp_launcher<<<1, 1, 0, stream>>>(nodes, result); },
                    stream, samples, 1, nodes, 1, result);
        Graph child(stream, nodes, false, out, true);
        cudaGraph_t parent;
        cudaGraphExec_t parent_exec;
        CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        graph_launcher<<<1, 1, 0, stream>>>(child.exec, result);
        CUDA(cudaStreamEndCapture(stream, &parent));
        CUDA(cudaGraphInstantiateWithFlags(&parent_exec, parent, 0));
        CUDA(cudaGraphUpload(parent_exec, stream));
        CUDA(cudaStreamSynchronize(stream));
        launch_case("device_graph_tail_" + std::to_string(nodes),
                    [&] { CUDA(cudaGraphLaunch(parent_exec, stream)); },
                    stream, samples, 1, nodes, 1, result);
        CUDA(cudaGraphExecDestroy(parent_exec));
        CUDA(cudaGraphDestroy(parent));
    }
    CUDA(cudaFree(result));
    CUDA(cudaFree(out));
}
template<int MODE>
__global__ void sync_kernel(int iterations, Result* results) {
    __shared__ volatile unsigned exchange[256];
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    unsigned lane = threadIdx.x;
    unsigned value = lane + 1;
    if constexpr (MODE == 4) {
        if (lane == 0) init(&barrier, blockDim.x);
    }
    __syncthreads();
    U64 start = clock64();
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile("add.u32 %0, %0, 1;" : "+r"(value));
        if constexpr (MODE == 1) __syncwarp();
        if constexpr (MODE == 2) __syncthreads();
        if constexpr (MODE == 3) asm volatile("bar.sync 1;" ::: "memory");
        if constexpr (MODE == 4) barrier.arrive_and_wait();
        if constexpr (MODE == 5) value = __shfl_xor_sync(0xffffffff, value, 1);
        if constexpr (MODE == 6) {
            exchange[lane] = value;
            __syncwarp();
            value = exchange[lane ^ 1];
            __syncwarp();
        }
    }
    U64 end = clock64();
    if (lane == 0) results[0] = {end - start, value, 0};
}

template<bool SYNC>
__global__ void cluster_kernel(int iterations, Result* results) {
    auto cluster = cg::this_cluster();
    unsigned value = threadIdx.x + 1;
    cluster.sync();
    U64 start = clock64();
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile("add.u32 %0, %0, 1;" : "+r"(value));
        if constexpr (SYNC) cluster.sync();
    }
    U64 end = clock64();
    if (threadIdx.x == 0) results[blockIdx.x] = {end - start, value, 0};
}

template<class Submit>
void cycle_case(const std::string& name, Submit submit, int samples, int iterations,
                int threads, int blocks, Result* result, cudaStream_t stream) {
    for (int i = 0; i < 3; ++i) submit();
    CUDA(cudaGetLastError());
    CUDA(cudaStreamSynchronize(stream));
    for (int sample = 0; sample < samples; ++sample) {
        submit();
        CUDA(cudaGetLastError());
        CUDA(cudaStreamSynchronize(stream));
        std::vector<Result> host(blocks);
        CUDA(cudaMemcpy(host.data(), result, blocks * sizeof(Result), cudaMemcpyDeviceToHost));
        U64 cycles = 0;
        for (auto r : host) {
            cycles = std::max(cycles, r.cycles);
            if (r.value != unsigned(iterations + 1)) throw std::runtime_error("sync checksum failed: " + name);
        }
        row("sync", name, sample, "cycles_per_iteration", double(cycles) / iterations,
            "cycles", iterations, 0, 0, threads, blocks);
    }
    std::cerr << "sync: " << name << " threads=" << threads << " complete\n";
}

void sync_suite(int samples, bool quick, cudaStream_t stream) {
    Result* result;
    CUDA(cudaMalloc(&result, 4 * sizeof(Result)));
    std::vector<int> lengths = quick ? std::vector<int>{1024} : std::vector<int>{1024, 4096, 16384};
    for (int threads : {32, 128, 256}) for (int iterations : lengths) {
        #define SYNC_CASE(mode, name) cycle_case(name, [&] { \
            sync_kernel<mode><<<1, threads, 0, stream>>>(iterations, result); \
            }, samples, iterations, threads, 1, result, stream)
        SYNC_CASE(0, "loop_control");
        SYNC_CASE(1, "syncwarp");
        SYNC_CASE(2, "syncthreads");
        SYNC_CASE(3, "named_barrier_1");
        SYNC_CASE(4, "mbarrier_arrive_wait");
        SYNC_CASE(5, "shuffle_xor");
        SYNC_CASE(6, "shared_exchange_two_warp_barriers");
        #undef SYNC_CASE
        for (int blocks : {2, 4}) {
            cudaLaunchConfig_t config{};
            config.gridDim = dim3(blocks);
            config.blockDim = dim3(threads);
            config.stream = stream;
            cudaLaunchAttribute attr{};
            attr.id = cudaLaunchAttributeClusterDimension;
            attr.val.clusterDim = {unsigned(blocks), 1, 1};
            config.attrs = &attr;
            config.numAttrs = 1;
            cycle_case("cluster_loop_control", [&] {
                CUDA(cudaLaunchKernelEx(&config, cluster_kernel<false>, iterations, result));
            }, samples, iterations, threads, blocks, result, stream);
            cycle_case("cluster_sync", [&] {
                CUDA(cudaLaunchKernelEx(&config, cluster_kernel<true>, iterations, result));
            }, samples, iterations, threads, blocks, result, stream);
        }
    }
    CUDA(cudaFree(result));
}
template<bool L1, bool VECTOR>
__device__ __forceinline__ unsigned load_next(const unsigned* address, unsigned& checksum) {
    unsigned x, y, z, w;
    if constexpr (VECTOR) {
        if constexpr (L1)
            asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(address) : "memory");
        else
            asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(address) : "memory");
        checksum ^= y ^ z ^ w;
    } else {
        if constexpr (L1)
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(address) : "memory");
        else
            asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(address) : "memory");
    }
    return x;
}

template<bool L1, bool VECTOR>
__global__ void chase_kernel(const unsigned* data, unsigned nodes, int steps, Result* out) {
    unsigned index = 0, checksum = 0;
    #pragma unroll 1
    for (unsigned i = 0; i < nodes; ++i) index = load_next<L1, VECTOR>(data + index, checksum);
    if (index != 0) { out->status = 1; return; }
    U64 start = clock64();
    #pragma unroll 1
    for (int i = 0; i < steps; ++i) index = load_next<L1, VECTOR>(data + index, checksum);
    __shared__ volatile unsigned sink;
    sink = index ^ checksum;
    __threadfence_block();
    U64 end = clock64();
    out->cycles = end - start;
    out->value = index;
    out->status = 0;
}

template<bool L1, bool VECTOR>
void memory_case(unsigned* data, const std::vector<unsigned>& host, unsigned nodes,
                 size_t bytes, int stride, bool random, int samples, int steps,
                 Result* result, cudaStream_t stream) {
    std::string name = std::string(L1 ? "ca_" : "cg_") + (VECTOR ? "v4_" : "scalar_") + (random ? "random" : "sequential");
    unsigned expected = 0;
    for (int i = 0; i < steps; ++i) expected = host[expected];
    chase_kernel<L1, VECTOR><<<1, 1, 0, stream>>>(data, nodes, steps, result);
    CUDA(cudaGetLastError());
    CUDA(cudaStreamSynchronize(stream));
    for (int sample = 0; sample < samples; ++sample) {
        chase_kernel<L1, VECTOR><<<1, 1, 0, stream>>>(data, nodes, steps, result);
        CUDA(cudaGetLastError());
        CUDA(cudaStreamSynchronize(stream));
        Result r;
        CUDA(cudaMemcpy(&r, result, sizeof(r), cudaMemcpyDeviceToHost));
        if (r.status || r.value != expected) throw std::runtime_error("pointer-chase validation failed");
        row("memory", name, sample, "cycles_per_dependent_load", double(r.cycles) / steps,
            "cycles", steps, bytes, stride, 1, nodes);
    }
    std::cerr << "memory: " << name << " span=" << bytes << " stride=" << stride << " complete\n";
}

void memory_suite(int samples, bool quick, cudaStream_t stream, const cudaDeviceProp& prop) {
    Result* result;
    CUDA(cudaMalloc(&result, sizeof(Result)));
    size_t l2 = prop.l2CacheSize;
    std::vector<size_t> sizes = quick ? std::vector<size_t>{16*1024, 1024*1024, 2*l2}
                                    : std::vector<size_t>{4*1024, 32*1024, 128*1024, 512*1024,
                                                          4*1024*1024, l2/4, l2/2, l2, 2*l2, 4*l2};
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    int steps = quick ? 16384 : 65536;
    for (size_t size : sizes) for (int stride : (quick ? std::vector<int>{128} : std::vector<int>{32, 128, 4096})) {
        size_t bytes = size / stride * stride;
        unsigned nodes = bytes / stride;
        if (nodes < 2 || bytes / sizeof(unsigned) > UINT32_MAX) continue;
        for (bool random : {false, true}) {
            if (quick && !random) continue;
            std::vector<unsigned> order(nodes);
            std::iota(order.begin(), order.end(), 0);
            std::mt19937 rng(2026);
            if (random) std::shuffle(order.begin(), order.end(), rng);
            std::vector<unsigned> host(bytes / sizeof(unsigned), 0);
            unsigned spacing = stride / sizeof(unsigned);
            for (unsigned i = 0; i < nodes; ++i) {
                unsigned at = order[i] * spacing;
                host[at] = order[(i + 1) % nodes] * spacing;
                host[at + 1] = order[i];
                host[at + 2] = order[i] ^ 0x12345678;
                host[at + 3] = order[i] ^ 0x87654321;
            }
            unsigned* data;
            CUDA(cudaMalloc(&data, bytes));
            CUDA(cudaMemcpyAsync(data, host.data(), bytes, cudaMemcpyHostToDevice, stream));
            CUDA(cudaStreamSynchronize(stream));
            memory_case<true, false>(data, host, nodes, bytes, stride, random, samples, steps, result, stream);
            memory_case<false, false>(data, host, nodes, bytes, stride, random, samples, steps, result, stream);
            memory_case<true, true>(data, host, nodes, bytes, stride, random, samples, steps, result, stream);
            memory_case<false, true>(data, host, nodes, bytes, stride, random, samples, steps, result, stream);
            CUDA(cudaFree(data));
        }
    }
    CUDA(cudaFree(result));
}

int main(int argc, char** argv) {
    try {
        std::string suite = argc > 1 ? argv[1] : "all";
        int samples = argc > 2 ? std::stoi(argv[2]) : 15;
        bool quick = argc > 3 && std::stoi(argv[3]) != 0;
        if (samples < 3 || samples > 1000 || (suite != "all" && suite != "launch" && suite != "sync" && suite != "memory"))
            throw std::runtime_error("usage: microbench [all|launch|sync|memory] [samples:3..1000] [quick:0|1]");
        CUDA(cudaSetDevice(0));
        cudaDeviceProp prop;
        CUDA(cudaGetDeviceProperties(&prop, 0));
        if (prop.major != 10) throw std::runtime_error("this binary targets SM100-family GPUs (B200)");
        std::cerr << "GPU=" << prop.name << " SMs=" << prop.multiProcessorCount
                  << " L2_bytes=" << prop.l2CacheSize << " shared_per_SM=" << prop.sharedMemPerMultiprocessor
                  << " runtime=" << CUDART_VERSION << " seed=2026 samples=" << samples << " quick=" << quick << '\n';
        std::cout << "suite,case,sample,metric,value,unit,iterations,bytes,stride,threads,nodes\n";
        cudaStream_t stream;
        CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        if (suite == "all" || suite == "launch") launch_suite(samples, quick, stream);
        if (suite == "all" || suite == "sync") sync_suite(samples, quick, stream);
        if (suite == "all" || suite == "memory") memory_suite(samples, quick, stream, prop);
        CUDA(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
