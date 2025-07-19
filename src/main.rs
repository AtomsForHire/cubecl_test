use cubecl::prelude::*;

/// Kernel for adding two 1D arrays, and placing result into output array.
#[cube(launch_unchecked)]
fn add_arrays<F: Float>(lhs: &Array<F>, rhs: &Array<F>, out: &mut Array<F>) {
    if UNIT_POS_X < lhs.len() {
        let mut sum = F::new(0.0);
        sum = lhs[UNIT_POS_X] + rhs[UNIT_POS_X];
        out[UNIT_POS_X] = sum;
    }
}

/// Set up client/context. Create data on cpu -> copy to gpu then run kernel
fn launch<R: Runtime, F: Float + CubeElement>(device: &R::Device) {
    let client = R::client(device);
    let lhs_cpu = &[-1., 10., 1., 5.];
    let rhs_cpu = &[-1., 10., 1., 5.];
    let len = lhs_cpu.len();

    // Create handles to data on the accelerator device
    let output_gpu = client.empty(len * core::mem::size_of::<f32>());
    let lhs_gpu = client.create(f32::as_bytes(lhs_cpu));
    let rhs_gpu = client.create(f32::as_bytes(rhs_cpu));

    unsafe {
        let array_arg_lhs = ArrayArg::from_raw_parts::<f32>(&lhs_gpu, len, 1);
        let array_arg_rhs = ArrayArg::from_raw_parts::<f32>(&rhs_gpu, len, 1);
        let array_arg_output = ArrayArg::from_raw_parts::<f32>(&output_gpu, len, 1);
        add_arrays::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),     // Create 1 cuboid
            CubeDim::new(len as u32, 1, 1), // Make cuboid one long row of, then we can run kernel
            // on each unit
            array_arg_lhs,
            array_arg_rhs,
            array_arg_output,
        );
    }

    // Read back data from gpu to cpu
    let bytes = client.read_one(output_gpu.clone().binding());
    let output_cpu = f32::from_bytes(&bytes);
    println!("{output_cpu:?}");
}

fn main() {
    println!("Hello, world!");
    launch::<cubecl::wgpu::WgpuRuntime, f32>(&Default::default());
}
