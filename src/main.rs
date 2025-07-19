use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn add_arrays<F: Float>(lhs: &Array<F>, rhs: &Array<F>, out: &mut Array<F>) {
    if UNIT_POS_X < lhs.len() {
        let mut sum = F::new(0.0);
        sum = lhs[UNIT_POS_X] + rhs[UNIT_POS_X];
        out[UNIT_POS_X] = sum;
    }
}

fn launch<R: Runtime, F: Float + CubeElement>(device: &R::Device) {
    let client = R::client(device);
    let lhs = &[-1., 10., 1., 5.];
    let rhs = &[-1., 10., 1., 5.];
    let len = lhs.len();

    // Create handles to data on the accelerator device
    let output = client.empty(len * core::mem::size_of::<f32>());
    let lhs = client.create(f32::as_bytes(lhs));
    let rhs = client.create(f32::as_bytes(rhs));

    unsafe {
        let array_lhs = ArrayArg::from_raw_parts::<f32>(&lhs, len, 1);
        let array_rhs = ArrayArg::from_raw_parts::<f32>(&rhs, len, 1);
        let array_output = ArrayArg::from_raw_parts::<f32>(&output, len, 1);
        add_arrays::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),     // Create 1 cuboid
            CubeDim::new(len as u32, 1, 1), // Make cuboid one long row of threads
            array_lhs,
            array_rhs,
            array_output,
        );
    }

    let bytes = client.read_one(output.clone().binding());
    let output = f32::from_bytes(&bytes);
    println!("{output:?}");
}

fn main() {
    println!("Hello, world!");
    launch::<cubecl::wgpu::WgpuRuntime, f32>(&Default::default());
}
