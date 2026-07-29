
import modal

from bench import benchmark_gemm

image = modal.Image.debian_slim(python_version="3.13").uv_pip_install("torch==2.11.0", "cutlass", "nvidia-cutlass", "nvidia-cutlass-dsl", "numpy").add_local_python_source("bench")
app = modal.App("milestone2", image=image)


@app.function(gpu="B200", timeout=1800)
def run():
    import cutlass.cute as cute
    import cutlass
    import numpy as np
    from cutlass.cute.runtime import from_dlpack


    @cute.struct
    class complex:
        real: cutlass.Float32
        imag: cutlass.Float32


    # SharedStorage size is 512, alignment is 128
    @cute.struct
    class SharedStorage:
        # struct elements with natural alignment
        a: cute.struct.MemRange[cutlass.Float32, 32]  # array
        b: cutlass.Int64  # scalar
        c: complex  # nested struct
        # struct elements with strict alignment
        x: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, 32],
            128,
        ]
        y: cute.struct.Align[cutlass.Int32, 8]
        z: cute.struct.Align[complex, 16]


    @cute.kernel
    def kernel(
        const_a: cutlass.Constexpr,
        dst_a: cute.Tensor,
        const_b: cutlass.Constexpr,
        dst_b: cute.Tensor,
        const_c: cutlass.Constexpr,
        dst_c: cute.Tensor,
    ):
        # Note: SMEM_SIZE bytes (specified in kernel().launch(smem=...)) can be reserved for developer to utilize
        # Note: alignment of initial allocator base ptr is 1024
        allocator = cutlass.utils.SmemAllocator()
        # base ptr of allocator points at: SMEM_ADDR_START (the starting address of available shared memory)

        # -- Allocate a scalar
        int_ptr = allocator.allocate(cutlass.Int32)
        # base ptr of allocator now points at: SMEM_ADDR_AFTER_INT = SMEM_ADDR_START + aligned_size(int)
        assert int_ptr.dtype == cutlass.Int32, "Expected Int32, but got {}".format(
            int_ptr.dtype
        )

        # -- Allocate a struct --
        # Note: when specified alignment, max(alignment, alignof(struct)) will be applied
        # reserves the section of struct in smem, elements in the struct can be accessed by ptr
        struct_in_smem = allocator.allocate(SharedStorage)
        # base ptr of allocator now points at: SMEM_ADDR_AFTER_STRUCT = SMEM_ADDR_START + aligned_size(struct)

        # -- Allocate a block of memory --
        # reserves a section of 64 bytes in smem, align to 128 bytes, returns the section base ptr
        section_in_smem = allocator.allocate(64, byte_alignment=128)
        # base ptr of allocator now points at: SMEM_ADDR_AFTER_SECTION = SMEM_ADDR_AFTER_STRUCT + aligned_size(section)

        # -- Allocate an array --
        # reserves an int64 array of size 14 in smem, returns the array base ptr
        array_in_smem = allocator.allocate_array(element_type=cutlass.Int64, num_elems=14)
        # base ptr of allocator now points at: SMEM_ADDR_AFTER_ARRAY = SMEM_ADDR_AFTER_SECTION + aligned_size(array)

        # -- Allocate a tensor --
        # Note: use cute.ComposedLayout or cute.Layout to specify layout of tensor
        # Note: iterator swizzle with swizzle layout is currently not supported
        layout = cute.make_layout((16, 2))
        tensor_in_smem = allocator.allocate_tensor(
            element_type=cutlass.Float32, layout=layout, byte_alignment=32, swizzle=None
        )
        # base ptr of allocator now points at: SMEM_ADDR_AFTER_TENSOR = SMEM_ADDR_AFTER_ARRAY + aligned_size(tensor)

        # ptr<f16, smem, align<1024>>
        # ptr<i64, smem, align<128>>
        # ptr<f32, smem, align<8>>
        print(struct_in_smem.a.data_ptr())
        print(struct_in_smem.b.ptr)
        print(struct_in_smem.c.real.ptr)
        # ptr<i8, smem, align<512>>
        print(section_in_smem)
        # ptr<i64, smem, align<64>>
        print(array_in_smem)
        # tensor<ptr<f16, smem, align<32>> o (16,4):(1,16)>
        print(tensor_in_smem)

        # assign struct member array element
        cute.printf("struct_in_smem.a[0] = {}", struct_in_smem.a[0])
        struct_in_smem.a[0] = 2
        cute.printf("struct_in_smem.a[0] = {}", struct_in_smem.a[0])

        # assign struct member scalar
        cute.printf("struct_in_smem.b.ptr = {}", struct_in_smem.b.ptr)
        cute.printf("struct_in_smem.b: value = {}", struct_in_smem.b.ptr.load())
        struct_in_smem.b = 16
        cute.printf("struct_in_smem.b: value = {}", struct_in_smem.b.ptr.load())

        # fill MemRange tensor in struct and copy to dst
        a_tensor = struct_in_smem.a.get_tensor(cute.make_layout((8, 4)))
        a_tensor.fill(const_a)
        cute.printf("cute.struct.MemRange: {}", a_tensor)
        dst_a.store(a_tensor.load())

        # convert block of smem to fill tensor and copy to dst
        layout = cute.make_layout((8, 2))
        sec_ptr = cute.recast_ptr(section_in_smem, dtype=cutlass.Float32)
        sec_tensor = cute.make_tensor(sec_ptr, layout)
        sec_tensor.fill(const_b)
        cute.printf("block of memory: {}", sec_tensor)
        dst_b.store(sec_tensor.load())

        # fill allocated tensor in smem and copy to dst
        tensor_in_smem.fill(const_c)
        cute.printf("tensor in smem: {}", tensor_in_smem)
        dst_c.store(tensor_in_smem.load())


    @cute.jit
    def host(
        const_a: cutlass.Constexpr,
        dst_a: cute.Tensor,
        const_b: cutlass.Constexpr,
        dst_b: cute.Tensor,
        const_c: cutlass.Constexpr,
        dst_c: cute.Tensor,
    ):
        # Note: Shared Memory size is automatically calculated now
        kernel(const_a, dst_a, const_b, dst_b, const_c, dst_c).launch(
            grid=(1, 1, 1),
            block=(1, 1, 1),
            # Automatically calculate the launch kernel shared memory usage when `smem=None`
        )


    def run_and_verify(const_a, const_b, const_c):
        import torch

        dst_a = torch.zeros((8, 4), dtype=torch.float32, device="cuda")
        dst_b = torch.zeros((8, 2), dtype=torch.float32, device="cuda")
        dst_c = torch.zeros((16, 2), dtype=torch.float32, device="cuda")

        host(
            const_a,
            from_dlpack(dst_a),
            const_b,
            from_dlpack(dst_b),
            const_c,
            from_dlpack(dst_c),
        )
        print("hello")

        assert const_a == dst_a.cpu()[0, 0], (f"Expected {const_a}, but got {dst_a.cpu()[0, 0]}")
        assert const_b == dst_b.cpu()[0, 0], (f"Expected {const_b}, but got {dst_b.cpu()[0, 0]}")
        assert const_c == dst_c.cpu()[0, 0], (f"Expected {const_c}, but got {dst_c.cpu()[0, 0]}")


    # prepare cuda context
    cutlass.cuda.initialize_cuda_context()
    # An example for shared memory allocation
    const_a = 0.5
    const_b = 1.0
    const_c = 2.0
    run_and_verify(const_a, const_b, const_c)
