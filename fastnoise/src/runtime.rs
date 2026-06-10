use crate::fastnoise_lite::{
    CellularDistanceFunction, CellularReturnType, DomainWarpType, FastNoiseLite, FractalType,
    NoiseType, RotationType3D,
};
use crate::model::{
    ChunkQuery, ChunkSummary, CombineOp, FastNoiseChange, FastNoiseCpuBackend, FastNoiseGraphSpec,
    FastNoiseKernel, FastNoisePlanner, FastNoisePlannerMeta, FastNoiseState, GraphDimension,
    NodeSpec, OutputKind, PositionSlots, PositionSource, SLOT_BASE_X, SLOT_BASE_Y, SLOT_BASE_Z,
    SLOT_DYNAMIC_START, SLOT_QUERY_F32, SLOT_QUERY_META, SLOT_QUERY_OFFSETS,
};
use braid::{
    BatchScratch, BraidError, BraidResult, BufferBinding, BufferSlot, BufferSpec, CancelFlag,
    CompiledPlan, ComputeScratch, CpuComputeBackend, CpuKernel, CpuKernelFactory, ElementKind,
    JobPacket, KernelSpec, PlannerBackend, PlannerScratch, SlotTable, StageSpec,
};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

const CANCEL_CHECK_EVERY_2D: usize = 4096;
const CANCEL_CHECK_MASK_2D: usize = CANCEL_CHECK_EVERY_2D - 1;
const CANCEL_CHECK_EVERY_3D: usize = 2048;
const CANCEL_CHECK_MASK_3D: usize = CANCEL_CHECK_EVERY_3D - 1;

pub fn make_cpu_backend() -> FastNoiseCpuBackend {
    let factory: Arc<dyn CpuKernelFactory> = Arc::new(FastNoiseKernelFactory);
    let mut backend = CpuComputeBackend::new();
    for kind in [
        FastNoiseKernel::InitGrid2d,
        FastNoiseKernel::InitGrid3d,
        FastNoiseKernel::Warp2d,
        FastNoiseKernel::Warp3d,
        FastNoiseKernel::Sample2d,
        FastNoiseKernel::Sample3d,
        FastNoiseKernel::Combine,
    ] {
        backend.register_factory(kind.kind(), Arc::clone(&factory));
    }
    backend
}

struct FastNoiseKernelFactory;

struct GridInitKernel {
    dimension: GraphDimension,
}

struct WarpKernel<const N: usize> {
    source: PositionSlots<N>,
    output: PositionSlots<N>,
    noise: Arc<FastNoiseLite>,
}

struct SampleKernel<const N: usize> {
    source: PositionSlots<N>,
    output: BufferSlot,
    noise: Arc<FastNoiseLite>,
}

struct CombineKernel {
    op: CombineOp,
    inputs: Vec<BufferSlot>,
    output: BufferSlot,
    params: Vec<f32>,
}

#[derive(Clone)]
struct WarpPayload<const N: usize> {
    source: PositionSlots<N>,
    output: PositionSlots<N>,
    noise: Arc<FastNoiseLite>,
}

#[derive(Clone)]
pub(crate) struct SamplePayload<const N: usize> {
    pub(crate) source: PositionSlots<N>,
    pub(crate) output: BufferSlot,
    pub(crate) noise: Arc<FastNoiseLite>,
}

#[derive(Clone)]
struct CombinePayload {
    op: CombineOp,
    inputs: Vec<BufferSlot>,
    output: BufferSlot,
    params: Vec<f32>,
}

pub(crate) trait KernelPayload: Sized {
    fn kind(&self) -> FastNoiseKernel;
    fn encode_into(&self, writer: &mut PayloadWriter<'_>) -> BraidResult<()>;
    fn decode_from(kind: FastNoiseKernel, reader: &mut PayloadReader<'_>) -> BraidResult<Self>;

    fn encode(&self, scratch: &mut PlannerScratch) -> BraidResult<KernelSpec> {
        scratch.reset();
        let mut writer = PayloadWriter::new(&mut scratch.bytes);
        self.encode_into(&mut writer)?;
        let payload = std::mem::take(&mut scratch.bytes);
        Ok(KernelSpec {
            kind_id: self.kind().kind(),
            payload: payload.into(),
            bindings: Vec::new(),
            dispatch: braid::DispatchHint::WholeBatch,
        })
    }

    fn decode(kind: FastNoiseKernel, bytes: &[u8]) -> BraidResult<Self> {
        let mut reader = PayloadReader::new(bytes);
        Self::decode_from(kind, &mut reader)
    }
}

impl PlannerBackend for FastNoisePlanner {
    type Spec = FastNoiseGraphSpec;
    type State = FastNoiseState;
    type Change = FastNoiseChange;
    type Query = ChunkQuery;
    type Resolution = ChunkSummary;
    type PlannerMeta = FastNoisePlannerMeta;
    const PREFER_ONE_QUERY_INLINE: bool = true;

    fn init_state(&self, spec: &Self::Spec) -> BraidResult<Self::State> {
        FastNoiseState::from_spec(spec)
    }

    fn reset_state(&self, state: &mut Self::State, spec: &Self::Spec) -> BraidResult<()> {
        state.reset_from_spec(spec)
    }

    fn apply(&self, state: &mut Self::State, changes: &[Self::Change]) -> BraidResult<()> {
        for change in changes {
            match change {
                FastNoiseChange::UpsertNode(node) => state.upsert_node(node.clone())?,
                FastNoiseChange::RemoveNode { id } => state.remove_node(id)?,
                FastNoiseChange::SetFinalField { id } => state.final_field = id.clone(),
            }
        }
        Ok(())
    }

    fn updated_state(
        &self,
        state: &Self::State,
        changes: &[Self::Change],
    ) -> BraidResult<Self::State> {
        let mut next = FastNoiseState {
            dimension: state.dimension,
            final_field: state.final_field.clone(),
            nodes: state.nodes.clone(),
            node_keys: state.node_keys.clone(),
        };
        self.apply(&mut next, changes)?;
        Ok(next)
    }

    fn compile(
        &self,
        state: &Self::State,
        scratch: &mut PlannerScratch,
    ) -> BraidResult<CompiledPlan<Self::PlannerMeta>> {
        compile_graph(state, scratch)
    }

    fn encode_batch(
        &self,
        plan: &CompiledPlan<Self::PlannerMeta>,
        queries: &[Self::Query],
        packet: &mut JobPacket,
        scratch: &mut BatchScratch,
    ) -> BraidResult<()> {
        scratch.reset();
        let query_count = queries.len();
        packet.query_count = query_count;

        let meta_values = {
            let meta_values = packet.ensure::<u32>(SLOT_QUERY_META, query_count * 3);
            unsafe { std::slice::from_raw_parts_mut(meta_values.as_mut_ptr(), meta_values.len()) }
        };
        let float_values = {
            let float_values = packet.ensure::<f32>(SLOT_QUERY_F32, query_count * 6);
            unsafe {
                std::slice::from_raw_parts_mut(float_values.as_mut_ptr(), float_values.len())
            };
        };
        let offset_values = {
            let offset_values = packet.ensure::<u32>(SLOT_QUERY_OFFSETS, query_count + 1);
            unsafe {
                std::slice::from_raw_parts_mut(offset_values.as_mut_ptr(), offset_values.len())
            }
        };
        let mut cursor = 0u64;
        offset_values[0] = 0;
        for (index, query) in queries.iter().enumerate() {
            let meta_base = index * 3;
            let float_base = index * 6;
            match (plan.planner_meta.dimension, query) {
                (
                    GraphDimension::D2,
                    ChunkQuery::Grid2D {
                        width,
                        height,
                        origin,
                        step,
                    },
                ) => {
                    let width_u32 = checked_u32_from_usize(*width, "query width")?;
                    let height_u32 = checked_u32_from_usize(*height, "query height")?;
                    meta_values[meta_base] = width_u32;
                    meta_values[meta_base + 1] = height_u32;
                    meta_values[meta_base + 2] = 1;
                    float_values[float_base] = origin[0];
                    float_values[float_base + 1] = origin[1];
                    float_values[float_base + 2] = 0.0;
                    float_values[float_base + 3] = step[0];
                    float_values[float_base + 4] = step[1];
                    float_values[float_base + 5] = 1.0;
                    cursor = checked_offset_after_add(cursor, sample_count_2d(*width, *height)?)?;
                }
                (
                    GraphDimension::D3,
                    ChunkQuery::Grid3D {
                        width,
                        height,
                        depth,
                        origin,
                        step,
                    },
                ) => {
                    let width_u32 = checked_u32_from_usize(*width, "query width")?;
                    let height_u32 = checked_u32_from_usize(*height, "query height")?;
                    let depth_u32 = checked_u32_from_usize(*depth, "query depth")?;
                    meta_values[meta_base] = width_u32;
                    meta_values[meta_base + 1] = height_u32;
                    meta_values[meta_base + 2] = depth_u32;
                    float_values[float_base] = origin[0];
                    float_values[float_base + 1] = origin[1];
                    float_values[float_base + 2] = origin[2];
                    float_values[float_base + 3] = step[0];
                    float_values[float_base + 4] = step[1];
                    float_values[float_base + 5] = step[2];
                    cursor = checked_offset_after_add(
                        cursor,
                        sample_count_3d(*width, *height, *depth)?,
                    )?;
                }
                (GraphDimension::D2, ChunkQuery::Grid3D { .. }) => unreachable!("2d query shape"),
                (GraphDimension::D3, ChunkQuery::Grid2D { .. }) => unreachable!("3d query shape"),
            }
            offset_values[index + 1] = cursor as u32;
        }

        Ok(())
    }

    fn encode_one(
        &self,
        plan: &CompiledPlan<Self::PlannerMeta>,
        query: &Self::Query,
        packet: &mut JobPacket,
        scratch: &mut BatchScratch,
    ) -> BraidResult<()> {
        scratch.reset();
        packet.query_count = 1;

        let (meta_ptr, meta_len) = {
            let meta_values = packet.ensure::<u32>(SLOT_QUERY_META, 3);
            let meta_len = meta_values.len();
            (meta_values.as_mut_ptr(), meta_len)
        };
        let meta_values = unsafe { std::slice::from_raw_parts_mut(meta_ptr, meta_len) };
        let (float_ptr, float_len) = {
            let float_values = packet.ensure::<f32>(SLOT_QUERY_F32, 6);
            let float_len = float_values.len();
            (float_values.as_mut_ptr(), float_len)
        };
        let float_values = unsafe { std::slice::from_raw_parts_mut(float_ptr, float_len) };
        let (offset_ptr, offset_len) = {
            let offset_values = packet.ensure::<u32>(SLOT_QUERY_OFFSETS, 2);
            let offset_len = offset_values.len();
            (offset_values.as_mut_ptr(), offset_len)
        };
        let offset_values = unsafe { std::slice::from_raw_parts_mut(offset_ptr, offset_len) };

        match (plan.planner_meta.dimension, query) {
            (
                GraphDimension::D2,
                ChunkQuery::Grid2D {
                    width,
                    height,
                    origin,
                    step,
                },
            ) => {
                meta_values[0] = *width as u32;
                meta_values[1] = *height as u32;
                meta_values[2] = 1;
                float_values[0] = origin[0];
                float_values[1] = origin[1];
                float_values[2] = 0.0;
                float_values[3] = step[0];
                float_values[4] = step[1];
                float_values[5] = 1.0;
                offset_values[0] = 0;
                offset_values[1] = sample_count_2d(*width, *height) as u32;
            }
            (
                GraphDimension::D3,
                ChunkQuery::Grid3D {
                    width,
                    height,
                    depth,
                    origin,
                    step,
                },
            ) => {
                meta_values[0] = *width as u32;
                meta_values[1] = *height as u32;
                meta_values[2] = *depth as u32;
                float_values[0] = origin[0];
                float_values[1] = origin[1];
                float_values[2] = origin[2];
                float_values[3] = step[0];
                float_values[4] = step[1];
                float_values[5] = step[2];
                offset_values[0] = 0;
                offset_values[1] = sample_count_3d(*width, *height, *depth) as u32;
            }
            (GraphDimension::D2, ChunkQuery::Grid3D { .. }) => unreachable!("2d query shape"),
            (GraphDimension::D3, ChunkQuery::Grid2D { .. }) => unreachable!("3d query shape"),
        }

        Ok(())
    }

    fn decode_batch(
        &self,
        plan: &CompiledPlan<Self::PlannerMeta>,
        packet: &JobPacket,
    ) -> BraidResult<Vec<Self::Resolution>> {
        let values = packet.slice::<f32>(plan.planner_meta.final_slot)?;
        let offsets = packet.slice::<u32>(SLOT_QUERY_OFFSETS)?;
        let mut summaries = Vec::with_capacity(offsets.len().saturating_sub(1));
        let mut start = 0usize;
        for end in offsets.iter().skip(1) {
            let end = *end as usize;
            summaries.push(summarize_samples(&values[start..end]));
            start = end;
        }
        Ok(summaries)
    }

    fn decode_one(
        &self,
        plan: &CompiledPlan<Self::PlannerMeta>,
        packet: &JobPacket,
    ) -> BraidResult<Self::Resolution> {
        let values = packet.slice::<f32>(plan.planner_meta.final_slot)?;
        let offsets = packet.slice::<u32>(SLOT_QUERY_OFFSETS)?;
        if offsets.len() < 2 {
            return Err(BraidError::from("missing fastnoise offsets"));
        }
        let end = offsets[1] as usize;
        let start = offsets[0] as usize;
        if values.len() < end || values.len() < start || start > end {
            return Err(BraidError::from("invalid fastnoise offsets"));
        }
        Ok(summarize_samples(&values[start..end]))
    }
}

impl FastNoiseState {
    fn from_spec(spec: &FastNoiseGraphSpec) -> BraidResult<Self> {
        let mut state = Self {
            dimension: spec.dimension,
            final_field: spec.final_field.clone(),
            nodes: SlotTable::default(),
            node_keys: HashMap::new(),
        };
        state.reset_from_spec(spec)?;
        Ok(state)
    }

    fn reset_from_spec(&mut self, spec: &FastNoiseGraphSpec) -> BraidResult<()> {
        self.dimension = spec.dimension;
        self.final_field = spec.final_field.clone();
        self.nodes.clear_reuse();
        self.node_keys.clear();
        for node in &spec.nodes {
            self.upsert_node(node.clone())?;
        }
        Ok(())
    }

    fn upsert_node(&mut self, node: NodeSpec) -> BraidResult<()> {
        if let Some(existing) = self.node_keys.get(node.id()).copied()
            && let Some(slot) = self.nodes.get_mut(existing)
        {
            *slot = node;
            return Ok(());
        }
        let id = node.id().to_owned();
        let key = self.nodes.insert(node);
        self.node_keys.insert(id, key);
        Ok(())
    }

    fn remove_node(&mut self, id: &str) -> BraidResult<()> {
        if let Some(key) = self.node_keys.remove(id) {
            self.nodes.remove(key);
        }
        Ok(())
    }
}

impl CpuKernelFactory for FastNoiseKernelFactory {
    fn prepare(
        &self,
        kernel: &KernelSpec,
        _scratch: &mut ComputeScratch,
    ) -> BraidResult<Box<dyn CpuKernel>> {
        let kind = FastNoiseKernel::from_kind(kernel.kind_id)
            .ok_or(BraidError::BackendRejectedKernel(kernel.kind_id))?;
        match kind {
            FastNoiseKernel::InitGrid2d => Ok(Box::new(GridInitKernel {
                dimension: GraphDimension::D2,
            })),
            FastNoiseKernel::InitGrid3d => Ok(Box::new(GridInitKernel {
                dimension: GraphDimension::D3,
            })),
            FastNoiseKernel::Warp2d | FastNoiseKernel::Warp3d => {
                if kind == FastNoiseKernel::Warp2d {
                    let payload = WarpPayload::<2>::decode(kind, &kernel.payload)?;
                    return Ok(Box::new(WarpKernel {
                        source: payload.source,
                        output: payload.output,
                        noise: payload.noise,
                    }));
                }
                let payload = WarpPayload::<3>::decode(kind, &kernel.payload)?;
                Ok(Box::new(WarpKernel {
                    source: payload.source,
                    output: payload.output,
                    noise: payload.noise,
                }))
            }
            FastNoiseKernel::Sample2d | FastNoiseKernel::Sample3d => {
                if kind == FastNoiseKernel::Sample2d {
                    let payload = SamplePayload::<2>::decode(kind, &kernel.payload)?;
                    return Ok(Box::new(SampleKernel {
                        source: payload.source,
                        output: payload.output,
                        noise: payload.noise,
                    }));
                }
                let payload = SamplePayload::<3>::decode(kind, &kernel.payload)?;
                Ok(Box::new(SampleKernel {
                    source: payload.source,
                    output: payload.output,
                    noise: payload.noise,
                }))
            }
            FastNoiseKernel::Combine => {
                let payload = CombinePayload::decode(kind, &kernel.payload)?;
                Ok(Box::new(CombineKernel {
                    op: payload.op,
                    inputs: payload.inputs,
                    output: payload.output,
                    params: payload.params,
                }))
            }
        }
    }
}

impl CpuKernel for GridInitKernel {
    fn run(&self, packet: &mut JobPacket, cancel: &CancelFlag) -> BraidResult<()> {
        let meta = {
            let meta = packet.slice::<u32>(SLOT_QUERY_META)?;
            unsafe { std::slice::from_raw_parts(meta.as_ptr(), meta.len()) }
        };
        let floats = {
            let floats = packet.slice::<f32>(SLOT_QUERY_F32)?;
            unsafe { std::slice::from_raw_parts(floats.as_ptr(), floats.len()) }
        };
        let offsets = {
            let offsets = packet.slice::<u32>(SLOT_QUERY_OFFSETS)?;
            unsafe { std::slice::from_raw_parts(offsets.as_ptr(), offsets.len()) }
        };
        let total = total_samples_from_offsets(offsets);
        let query_count = packet.query_count;
        packet.ensure::<f32>(SLOT_BASE_X, total);
        packet.ensure::<f32>(SLOT_BASE_Y, total);
        if self.dimension == GraphDimension::D3 {
            packet.ensure::<f32>(SLOT_BASE_Z, total);
            return packet.with_slices::<f32, _, _>(
                [SLOT_BASE_X, SLOT_BASE_Y, SLOT_BASE_Z],
                |buffers| {
                    let [xs, ys, zs] = buffers;
                    for (query_index, offset_value) in
                        offsets.iter().take(query_count).copied().enumerate()
                    {
                        let meta_base = query_index * 3;
                        let float_base = query_index * 6;
                        let width = meta[meta_base] as usize;
                        let height = meta[meta_base + 1] as usize;
                        let depth = meta[meta_base + 2] as usize;
                        let offset = offset_value as usize;
                        let origin_x = floats[float_base];
                        let origin_y = floats[float_base + 1];
                        let origin_z = floats[float_base + 2];
                        let step_x = floats[float_base + 3];
                        let step_y = floats[float_base + 4];
                        let step_z = floats[float_base + 5];
                        for z in 0..depth {
                            for y in 0..height {
                                for x in 0..width {
                                    let index = offset + ((z * height + y) * width) + x;
                                    xs[index] = origin_x + (x as f32 * step_x);
                                    ys[index] = origin_y + (y as f32 * step_y);
                                    zs[index] = origin_z + (z as f32 * step_z);
                                }
                                if y & 15 == 0 && cancel.is_cancelled() {
                                    return Err(BraidError::Cancelled);
                                }
                            }
                        }
                    }
                    Ok(())
                },
            );
        }
        packet.with_slices::<f32, _, _>([SLOT_BASE_X, SLOT_BASE_Y], |buffers| {
            let [xs, ys] = buffers;
            for (query_index, offset_value) in offsets.iter().take(query_count).copied().enumerate()
            {
                let meta_base = query_index * 3;
                let float_base = query_index * 6;
                let width = meta[meta_base] as usize;
                let height = meta[meta_base + 1] as usize;
                let offset = offset_value as usize;
                let origin_x = floats[float_base];
                let origin_y = floats[float_base + 1];
                let step_x = floats[float_base + 3];
                let step_y = floats[float_base + 4];
                for y in 0..height {
                    for x in 0..width {
                        let index = offset + (y * width) + x;
                        xs[index] = origin_x + (x as f32 * step_x);
                        ys[index] = origin_y + (y as f32 * step_y);
                    }
                    if y & 31 == 0 && cancel.is_cancelled() {
                        return Err(BraidError::Cancelled);
                    }
                }
            }
            Ok(())
        })
    }
}

impl CpuKernel for WarpKernel<2> {
    fn run(&self, packet: &mut JobPacket, cancel: &CancelFlag) -> BraidResult<()> {
        let total = total_samples_from_offsets(packet.slice::<u32>(SLOT_QUERY_OFFSETS)?);
        let [source_x, source_y] = self.source.coords;
        let [output_x, output_y] = self.output.coords;
        packet.ensure::<f32>(output_x, total);
        packet.ensure::<f32>(output_y, total);
        packet.with_slices::<f32, _, _>([source_x, source_y, output_x, output_y], |buffers| {
            let [xs, ys, out_xs, out_ys] = buffers;
            for index in 0..out_xs.len() {
                let (warp_x, warp_y) = self.noise.domain_warp_2d(xs[index], ys[index]);
                out_xs[index] = warp_x;
                out_ys[index] = warp_y;
                if index & CANCEL_CHECK_MASK_2D == 0 && cancel.is_cancelled() {
                    return Err(BraidError::Cancelled);
                }
            }
            Ok(())
        })
    }
}

impl CpuKernel for WarpKernel<3> {
    fn run(&self, packet: &mut JobPacket, cancel: &CancelFlag) -> BraidResult<()> {
        let total = total_samples_from_offsets(packet.slice::<u32>(SLOT_QUERY_OFFSETS)?);
        let [source_x, source_y, source_z] = self.source.coords;
        let [output_x, output_y, output_z] = self.output.coords;
        packet.ensure::<f32>(output_x, total);
        packet.ensure::<f32>(output_y, total);
        packet.ensure::<f32>(output_z, total);
        packet.with_slices::<f32, _, _>(
            [source_x, source_y, source_z, output_x, output_y, output_z],
            |buffers| {
                let [xs, ys, zs, out_xs, out_ys, out_zs] = buffers;
                for index in 0..out_xs.len() {
                    let (warp_x, warp_y, warp_z) =
                        self.noise.domain_warp_3d(xs[index], ys[index], zs[index]);
                    out_xs[index] = warp_x;
                    out_ys[index] = warp_y;
                    out_zs[index] = warp_z;
                    if index & CANCEL_CHECK_MASK_3D == 0 && cancel.is_cancelled() {
                        return Err(BraidError::Cancelled);
                    }
                }
                Ok(())
            },
        )
    }
}

impl CpuKernel for SampleKernel<2> {
    fn run(&self, packet: &mut JobPacket, cancel: &CancelFlag) -> BraidResult<()> {
        let total = total_samples_from_offsets(packet.slice::<u32>(SLOT_QUERY_OFFSETS)?);
        let [source_x, source_y] = self.source.coords;
        packet.ensure::<f32>(self.output, total);
        packet.with_slices::<f32, _, _>([source_x, source_y, self.output], |buffers| {
            let [xs, ys, out] = buffers;
            for index in 0..out.len() {
                out[index] = self.noise.get_noise_2d(xs[index], ys[index]);
                if index & CANCEL_CHECK_MASK_2D == 0 && cancel.is_cancelled() {
                    return Err(BraidError::Cancelled);
                }
            }
            Ok(())
        })
    }
}

impl CpuKernel for SampleKernel<3> {
    fn run(&self, packet: &mut JobPacket, cancel: &CancelFlag) -> BraidResult<()> {
        let total = total_samples_from_offsets(packet.slice::<u32>(SLOT_QUERY_OFFSETS)?);
        let [source_x, source_y, source_z] = self.source.coords;
        packet.ensure::<f32>(self.output, total);
        packet.with_slices::<f32, _, _>([source_x, source_y, source_z, self.output], |buffers| {
            let [xs, ys, zs, out] = buffers;
            for index in 0..out.len() {
                out[index] = self.noise.get_noise_3d(xs[index], ys[index], zs[index]);
                if index & CANCEL_CHECK_MASK_3D == 0 && cancel.is_cancelled() {
                    return Err(BraidError::Cancelled);
                }
            }
            Ok(())
        })
    }
}

impl CpuKernel for CombineKernel {
    fn run(&self, packet: &mut JobPacket, cancel: &CancelFlag) -> BraidResult<()> {
        let total = total_samples_from_offsets(packet.slice::<u32>(SLOT_QUERY_OFFSETS)?);
        packet.ensure::<f32>(self.output, total);
        match self.op {
            CombineOp::Add => self.run_binary(packet, cancel, |a, b| a + b),
            CombineOp::Sub => self.run_binary(packet, cancel, |a, b| a - b),
            CombineOp::Mul => self.run_binary(packet, cancel, |a, b| a * b),
            CombineOp::Min => self.run_binary(packet, cancel, |a, b| a.min(b)),
            CombineOp::Max => self.run_binary(packet, cancel, |a, b| a.max(b)),
            CombineOp::Clamp => {
                let [min_value, max_value] = expect_params(self.params.as_slice(), self.op)?;
                let input = expect_input(self.inputs.as_slice())?;
                packet.with_slices::<f32, _, _>([input, self.output], |buffers| {
                    let [values, out] = buffers;
                    for index in 0..out.len() {
                        out[index] = values[index].clamp(min_value, max_value);
                        if index & CANCEL_CHECK_MASK_2D == 0 && cancel.is_cancelled() {
                            return Err(BraidError::Cancelled);
                        }
                    }
                    Ok(())
                })
            }
            CombineOp::Remap => {
                let [src_min, src_max, dst_min, dst_max] =
                    expect_params(self.params.as_slice(), self.op)?;
                let input = expect_input(self.inputs.as_slice())?;
                packet.with_slices::<f32, _, _>([input, self.output], |buffers| {
                    let [values, out] = buffers;
                    for index in 0..out.len() {
                        let denom = src_max - src_min;
                        let t = if denom.abs() <= f32::EPSILON {
                            0.0
                        } else {
                            ((values[index] - src_min) / denom).clamp(0.0, 1.0)
                        };
                        out[index] = dst_min + ((dst_max - dst_min) * t);
                        if index & CANCEL_CHECK_MASK_2D == 0 && cancel.is_cancelled() {
                            return Err(BraidError::Cancelled);
                        }
                    }
                    Ok(())
                })
            }
            CombineOp::YGradient => {
                let [y_min, y_max, out_min, out_max] =
                    expect_params(self.params.as_slice(), self.op)?;
                let input = expect_input(self.inputs.as_slice())?;
                packet.with_slices::<f32, _, _>([input, SLOT_BASE_Y, self.output], |buffers| {
                    let [values, ys, out] = buffers;
                    for index in 0..out.len() {
                        let denom = y_max - y_min;
                        let t = if denom.abs() <= f32::EPSILON {
                            0.0
                        } else {
                            ((ys[index] - y_min) / denom).clamp(0.0, 1.0)
                        };
                        let gradient = out_min + ((out_max - out_min) * t);
                        out[index] = values[index] + gradient;
                        if index & CANCEL_CHECK_MASK_3D == 0 && cancel.is_cancelled() {
                            return Err(BraidError::Cancelled);
                        }
                    }
                    Ok(())
                })
            }
        }
    }
}

impl CombineKernel {
    fn run_binary(
        &self,
        packet: &mut JobPacket,
        cancel: &CancelFlag,
        op: impl Fn(f32, f32) -> f32,
    ) -> BraidResult<()> {
        let (left, right) = expect_two_inputs(self.inputs.as_slice())?;
        packet.with_slices::<f32, _, _>([left, right, self.output], |buffers| {
            let [lhs, rhs, out] = buffers;
            for index in 0..out.len() {
                out[index] = op(lhs[index], rhs[index]);
                if index & CANCEL_CHECK_MASK_2D == 0 && cancel.is_cancelled() {
                    return Err(BraidError::Cancelled);
                }
            }
            Ok(())
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct NodeHandle(usize);

impl NodeHandle {
    fn index(self) -> usize {
        self.0
    }
}

struct CompileNodes<'a> {
    nodes: Vec<&'a NodeSpec>,
    handles_by_id: HashMap<&'a str, NodeHandle>,
    outputs: Vec<OutputKind>,
}

impl<'a> CompileNodes<'a> {
    fn build(state: &'a FastNoiseState) -> BraidResult<Self> {
        let mut nodes = Vec::with_capacity(state.nodes.len());
        let mut handles_by_id = HashMap::with_capacity(state.nodes.len());
        for (_, node) in state.nodes.iter() {
            let handle = NodeHandle(nodes.len());
            handles_by_id.insert(node.id(), handle);
            nodes.push(node);
        }
        let outputs = nodes
            .iter()
            .map(|node| node.output_kind(state.dimension))
            .collect();
        Ok(Self {
            nodes,
            handles_by_id,
            outputs,
        })
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn handles(&self) -> impl Iterator<Item = NodeHandle> + '_ {
        (0..self.nodes.len()).map(NodeHandle)
    }

    fn node(&self, handle: NodeHandle) -> &'a NodeSpec {
        self.nodes[handle.index()]
    }

    fn output(&self, handle: NodeHandle) -> OutputKind {
        self.outputs[handle.index()]
    }

    fn resolve(&self, id: &str) -> Option<NodeHandle> {
        self.handles_by_id.get(id).copied()
    }
}

fn compile_graph(
    state: &FastNoiseState,
    scratch: &mut PlannerScratch,
) -> BraidResult<CompiledPlan<FastNoisePlannerMeta>> {
    let graph = CompileNodes::build(state)?;
    let mut indegree = vec![0usize; graph.len()];
    let mut adjacency = vec![Vec::new(); graph.len()];
    for handle in graph.handles() {
        let deps = collect_dependencies(&graph, handle)?;
        for dep in deps {
            adjacency[dep.index()].push(handle);
            indegree[handle.index()] += 1;
        }
    }

    let final_handle = resolve_node(&graph, state.final_field.as_str())?;

    let mut queue = VecDeque::new();
    for handle in graph.handles() {
        if indegree[handle.index()] == 0 {
            queue.push_back(handle);
        }
    }
    let mut sorted = Vec::with_capacity(graph.len());
    while let Some(handle) = queue.pop_front() {
        sorted.push(handle);
        for child in adjacency[handle.index()].iter().copied() {
            let entry = &mut indegree[child.index()];
            *entry = entry.saturating_sub(1);
            if *entry == 0 {
                queue.push_back(child);
            }
        }
    }
    if sorted.len() != graph.len() {
        return Err(BraidError::InvalidSpec(
            "cycle detected in fastnoise graph".to_owned(),
        ));
    }

    let mut next_slot = SLOT_DYNAMIC_START;
    let mut position_slots_2 = vec![None; graph.len()];
    let mut position_slots_3 = vec![None; graph.len()];
    let mut scalar_slots = vec![None; graph.len()];
    let mut buffers = vec![
        BufferSpec {
            slot: SLOT_QUERY_META,
            element_kind: ElementKind::U32,
            layout: braid::BufferLayout::Dynamic,
        },
        BufferSpec {
            slot: SLOT_QUERY_F32,
            element_kind: ElementKind::F32,
            layout: braid::BufferLayout::Dynamic,
        },
        BufferSpec {
            slot: SLOT_QUERY_OFFSETS,
            element_kind: ElementKind::U32,
            layout: braid::BufferLayout::Dynamic,
        },
        BufferSpec {
            slot: SLOT_BASE_X,
            element_kind: ElementKind::F32,
            layout: braid::BufferLayout::Dynamic,
        },
        BufferSpec {
            slot: SLOT_BASE_Y,
            element_kind: ElementKind::F32,
            layout: braid::BufferLayout::Dynamic,
        },
    ];
    if state.dimension == GraphDimension::D3 {
        buffers.push(BufferSpec {
            slot: SLOT_BASE_Z,
            element_kind: ElementKind::F32,
            layout: braid::BufferLayout::Dynamic,
        });
    }

    for handle in sorted.iter().copied() {
        match graph.output(handle) {
            OutputKind::Position(GraphDimension::D2) => {
                let slots = allocate_position_slots::<2>(&mut next_slot);
                position_slots_2[handle.index()] = Some(slots);
                for slot in slots.coords {
                    buffers.push(dynamic_f32_buffer(slot));
                }
            }
            OutputKind::Position(GraphDimension::D3) => {
                let slots = allocate_position_slots::<3>(&mut next_slot);
                position_slots_3[handle.index()] = Some(slots);
                for slot in slots.coords {
                    buffers.push(dynamic_f32_buffer(slot));
                }
            }
            OutputKind::Scalar(_) => {
                let slot = allocate_slot(&mut next_slot);
                scalar_slots[handle.index()] = Some(slot);
                buffers.push(dynamic_f32_buffer(slot));
            }
        }
    }

    let mut stages = Vec::with_capacity(sorted.len() + 1);
    stages.push(StageSpec {
        kernels: vec![KernelSpec {
            kind_id: match state.dimension {
                GraphDimension::D2 => FastNoiseKernel::InitGrid2d.kind(),
                GraphDimension::D3 => FastNoiseKernel::InitGrid3d.kind(),
            },
            payload: Arc::from([]),
            bindings: vec![
                BufferBinding {
                    slot: SLOT_QUERY_META,
                    access: braid::BufferAccess::Read,
                },
                BufferBinding {
                    slot: SLOT_QUERY_F32,
                    access: braid::BufferAccess::Read,
                },
                BufferBinding {
                    slot: SLOT_QUERY_OFFSETS,
                    access: braid::BufferAccess::Read,
                },
            ],
            dispatch: braid::DispatchHint::WholeBatch,
        }],
    });

    for handle in sorted.iter().copied() {
        let node = graph.node(handle);
        let kernel = match node {
            NodeSpec::Warp2D(node) => WarpPayload::<2> {
                source: resolve_position_source(
                    &node.source,
                    &graph,
                    &position_slots_2,
                    base_position_slots_2(),
                )?,
                output: expect_position_slots(position_slots_2.as_slice(), handle)?,
                noise: node.noise.clone(),
            }
            .encode(scratch),
            NodeSpec::Warp3D(node) => WarpPayload::<3> {
                source: resolve_position_source(
                    &node.source,
                    &graph,
                    &position_slots_3,
                    base_position_slots_3(),
                )?,
                output: expect_position_slots(position_slots_3.as_slice(), handle)?,
                noise: node.noise.clone(),
            }
            .encode(scratch),
            NodeSpec::Sample2D(node) => SamplePayload::<2> {
                source: resolve_position_source(
                    &node.source,
                    &graph,
                    &position_slots_2,
                    base_position_slots_2(),
                )?,
                output: expect_scalar_slot(scalar_slots.as_slice(), handle)?,
                noise: node.noise.clone(),
            }
            .encode(scratch),
            NodeSpec::Sample3D(node) => SamplePayload::<3> {
                source: resolve_position_source(
                    &node.source,
                    &graph,
                    &position_slots_3,
                    base_position_slots_3(),
                )?,
                output: expect_scalar_slot(scalar_slots.as_slice(), handle)?,
                noise: node.noise.clone(),
            }
            .encode(scratch),
            NodeSpec::Combine(node) => {
                validate_combine_contract(node.op, node.inputs.len(), node.params.len())?;
                let mut inputs = Vec::with_capacity(node.inputs.len());
                for input in &node.inputs {
                    let input_handle = resolve_node(&graph, input.as_str())?;
                    inputs.push(expect_scalar_slot(scalar_slots.as_slice(), input_handle)?);
                }
                CombinePayload {
                    op: node.op,
                    inputs,
                    output: expect_scalar_slot(scalar_slots.as_slice(), handle)?,
                    params: node.params.clone(),
                }
                .encode(scratch)
            }
        }?;
        stages.push(StageSpec {
            kernels: vec![kernel],
        });
    }

    Ok(CompiledPlan {
        pipeline: braid::PipelineShape { buffers, stages },
        static_buffers: Vec::new(),
        planner_meta: FastNoisePlannerMeta {
            dimension: state.dimension,
            final_slot: expect_scalar_slot(scalar_slots.as_slice(), final_handle)?,
        },
    })
}

fn collect_dependencies(
    graph: &CompileNodes<'_>,
    handle: NodeHandle,
) -> BraidResult<Vec<NodeHandle>> {
    match graph.node(handle) {
        NodeSpec::Warp2D(node) => Ok(position_source_dependency(&node.source, graph)?
            .into_iter()
            .collect()),
        NodeSpec::Warp3D(node) => Ok(position_source_dependency(&node.source, graph)?
            .into_iter()
            .collect()),
        NodeSpec::Sample2D(node) => Ok(position_source_dependency(&node.source, graph)?
            .into_iter()
            .collect()),
        NodeSpec::Sample3D(node) => Ok(position_source_dependency(&node.source, graph)?
            .into_iter()
            .collect()),
        NodeSpec::Combine(node) => node
            .inputs
            .iter()
            .map(|input| resolve_node(graph, input.as_str()))
            .collect(),
    }
}

fn position_source_dependency(
    source: &PositionSource,
    graph: &CompileNodes<'_>,
) -> BraidResult<Option<NodeHandle>> {
    match source {
        PositionSource::Base => Ok(None),
        PositionSource::Node(id) => Ok(Some(resolve_node(graph, id.as_str())?)),
    }
}

fn resolve_position_source<const N: usize>(
    source: &PositionSource,
    graph: &CompileNodes<'_>,
    position_slots: &[Option<PositionSlots<N>>],
    base_slots: PositionSlots<N>,
) -> BraidResult<PositionSlots<N>> {
    match source {
        PositionSource::Base => Ok(base_slots),
        PositionSource::Node(id) => {
            let handle = resolve_node(graph, id.as_str())?;
            expect_position_slots(position_slots, handle)
        }
    }
}

fn allocate_slot(next_slot: &mut u16) -> BufferSlot {
    let slot = BufferSlot(*next_slot);
    *next_slot += 1;
    slot
}

fn base_position_slots_2() -> PositionSlots<2> {
    PositionSlots {
        coords: [SLOT_BASE_X, SLOT_BASE_Y],
    }
}

fn base_position_slots_3() -> PositionSlots<3> {
    PositionSlots {
        coords: [SLOT_BASE_X, SLOT_BASE_Y, SLOT_BASE_Z],
    }
}

fn allocate_position_slots<const N: usize>(next_slot: &mut u16) -> PositionSlots<N> {
    let mut coords = [BufferSlot(0); N];
    for slot in &mut coords {
        *slot = allocate_slot(next_slot);
    }
    PositionSlots { coords }
}

fn dynamic_f32_buffer(slot: BufferSlot) -> BufferSpec {
    BufferSpec {
        slot,
        element_kind: ElementKind::F32,
        layout: braid::BufferLayout::Dynamic,
    }
}

pub(crate) fn summarize_samples(values: &[f32]) -> ChunkSummary {
    if values.is_empty() {
        return ChunkSummary {
            samples: 0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            checksum: 0,
            taps: [0.0; 8],
        };
    }

    let mut min = values[0];
    let mut max = values[0];
    let mut sum = 0.0f64;
    let mut checksum = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        min = min.min(*value);
        max = max.max(*value);
        sum += f64::from(*value);
        checksum ^= u64::from(value.to_bits()) + 0x9e37_79b9;
        checksum = checksum.wrapping_mul(0x1000_0000_01b3);
    }

    let mut taps = [0.0; 8];
    let tap_count = taps.len();
    for (index, tap) in taps.iter_mut().enumerate() {
        let sample_index = if values.len() == 1 {
            0
        } else {
            index * (values.len() - 1) / (tap_count - 1)
        };
        *tap = values[sample_index];
    }

    ChunkSummary {
        samples: values.len(),
        min,
        max,
        mean: (sum / values.len() as f64) as f32,
        checksum,
        taps,
    }
}

fn sample_count_2d(width: usize, height: usize) -> BraidResult<u32> {
    let width = checked_u32_from_usize(width, "query width")?;
    let height = checked_u32_from_usize(height, "query height")?;
    checked_mul_u32(width, height, "query sample count")
}

fn sample_count_3d(width: usize, height: usize, depth: usize) -> BraidResult<u32> {
    let width = checked_u32_from_usize(width, "query width")?;
    let height = checked_u32_from_usize(height, "query height")?;
    let depth = checked_u32_from_usize(depth, "query depth")?;
    checked_mul_u32(width, height, "intermediate query sample count")?
        .checked_mul(depth)
        .ok_or_else(|| BraidError::InvalidSpec("query sample count exceeds u32".to_owned()))
}

fn checked_u32_from_usize(value: usize, field: &str) -> BraidResult<u32> {
    u32::try_from(value).map_err(|_| {
        BraidError::InvalidSpec(format!("{field} exceeds u32 metadata range: {value}"))
    })
}

fn checked_mul_u32(left: u32, right: u32, label: &str) -> BraidResult<u32> {
    let left = left as u64;
    let right = right as u64;
    let product = left
        .checked_mul(right)
        .ok_or_else(|| BraidError::InvalidSpec(format!("{label} overflow")))?;
    if product > u32::MAX as u64 {
        return Err(BraidError::InvalidSpec(format!(
            "{label} exceeds u32 metadata range: {product}"
        )));
    }
    Ok(product as u32)
}

fn checked_offset_after_add(offset: u64, add: u32) -> BraidResult<u64> {
    let next = offset
        .checked_add(add as u64)
        .ok_or_else(|| BraidError::InvalidSpec("chunk sample offset overflowed".to_owned()))?;
    if next > u32::MAX as u64 {
        return Err(BraidError::InvalidSpec(
            "chunk sample offset exceeds u32 metadata range".to_owned(),
        ));
    }
    Ok(next)
}

fn total_samples_from_offsets(offsets: &[u32]) -> usize {
    offsets.last().copied().unwrap_or(0) as usize
}

fn resolve_node(graph: &CompileNodes<'_>, id: &str) -> BraidResult<NodeHandle> {
    let Some(handle) = graph.resolve(id) else {
        return Err(BraidError::InvalidSpec(format!("missing node '{}'", id)));
    };
    Ok(handle)
}

fn expect_position_slots<const N: usize>(
    slots: &[Option<PositionSlots<N>>],
    handle: NodeHandle,
) -> BraidResult<PositionSlots<N>> {
    let Some(slots) = slots[handle.index()] else {
        return Err(BraidError::InvalidSpec("missing position slots".to_owned()));
    };
    Ok(slots)
}

fn expect_scalar_slot(slots: &[Option<BufferSlot>], handle: NodeHandle) -> BraidResult<BufferSlot> {
    let Some(slot) = slots[handle.index()] else {
        return Err(BraidError::InvalidSpec("missing scalar slot".to_owned()));
    };
    Ok(slot)
}

fn expect_two_inputs(inputs: &[BufferSlot]) -> BraidResult<(BufferSlot, BufferSlot)> {
    let mut iter = inputs.iter();
    let left = *iter.next().ok_or_else(|| {
        BraidError::InvalidSpec("combine op expects 2 inputs, but got none".to_owned())
    })?;
    let right = *iter.next().ok_or_else(|| {
        BraidError::InvalidSpec("combine op expects 2 inputs, but got only 1".to_owned())
    })?;
    if iter.next().is_some() {
        return Err(BraidError::InvalidSpec(
            "combine op expects 2 inputs, but got more than 2".to_owned(),
        ));
    }
    Ok((left, right))
}

fn expect_input(inputs: &[BufferSlot]) -> BraidResult<BufferSlot> {
    let mut iter = inputs.iter();
    let input = *iter.next().ok_or_else(|| {
        BraidError::InvalidSpec("combine op expects 1 input, but got none".to_owned())
    })?;
    if iter.next().is_some() {
        return Err(BraidError::InvalidSpec(
            "combine op expects 1 input, but got more than 1".to_owned(),
        ));
    }
    Ok(input)
}

fn expect_params<const N: usize>(params: &[f32], op: CombineOp) -> BraidResult<[f32; N]> {
    let mut out = [0.0; N];
    if params.len() != N {
        return Err(BraidError::InvalidSpec(format!(
            "combine op {:?} expects {N} parameters, got {}",
            op,
            params.len()
        )));
    }
    out.copy_from_slice(&params[..N]);
    Ok(out)
}

fn validate_combine_contract(
    op: CombineOp,
    input_count: usize,
    param_count: usize,
) -> BraidResult<()> {
    let (expected_inputs, expected_params) = match op {
        CombineOp::Add | CombineOp::Sub | CombineOp::Mul | CombineOp::Min | CombineOp::Max => {
            (2, 0)
        }
        CombineOp::Clamp => (1, 2),
        CombineOp::Remap | CombineOp::YGradient => (1, 4),
    };
    if input_count != expected_inputs {
        return Err(BraidError::InvalidSpec(format!(
            "combine op {op:?} expects {expected_inputs} inputs, got {input_count}"
        )));
    }
    if param_count != expected_params {
        return Err(BraidError::InvalidSpec(format!(
            "combine op {op:?} expects {expected_params} parameters, got {param_count}"
        )));
    }
    Ok(())
}

macro_rules! enum_codec {
    ($encode:ident, $decode:ident, $ty:ty, $label:literal {
        $($variant:path => $tag:expr),+ $(,)?
    }) => {
        fn $encode(value: $ty) -> u32 {
            match value {
                $($variant => $tag,)+
            }
        }

        fn $decode(value: u32) -> BraidResult<$ty> {
            match value {
                $($tag => Ok($variant),)+
                _ => Err(BraidError::InvalidSpec(format!(
                    "unknown {} {}",
                    $label, value
                ))),
            }
        }
    };
}

pub(crate) struct PayloadWriter<'a> {
    bytes: &'a mut Vec<u8>,
}

impl<'a> PayloadWriter<'a> {
    fn new(bytes: &'a mut Vec<u8>) -> Self {
        Self { bytes }
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn slot(&mut self, value: BufferSlot) {
        self.u16(value.0);
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn i32(&mut self, value: i32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn f32(&mut self, value: f32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn noise(&mut self, noise: &FastNoiseLite) {
        self.i32(noise.seed);
        self.f32(noise.frequency);
        self.u32(encode_noise_type(noise.noise_type));
        self.u32(encode_rotation_type(noise.rotation_type_3d));
        self.u32(encode_fractal_type(noise.fractal_type));
        self.i32(noise.octaves);
        self.f32(noise.lacunarity);
        self.f32(noise.gain);
        self.f32(noise.weighted_strength);
        self.f32(noise.ping_pong_strength);
        self.u32(encode_cellular_distance_function(
            noise.cellular_distance_function,
        ));
        self.u32(encode_cellular_return_type(noise.cellular_return_type));
        self.f32(noise.cellular_jitter_modifier);
        self.u32(encode_domain_warp_type(noise.domain_warp_type));
        self.f32(noise.domain_warp_amp);
    }
}

enum_codec!(encode_noise_type, decode_noise_type, NoiseType, "noise type tag" {
    NoiseType::OpenSimplex2 => 0,
    NoiseType::OpenSimplex2S => 1,
    NoiseType::Cellular => 2,
    NoiseType::Perlin => 3,
    NoiseType::ValueCubic => 4,
    NoiseType::Value => 5,
});

enum_codec!(encode_rotation_type, decode_rotation_type, RotationType3D, "rotation type tag" {
    RotationType3D::None => 0,
    RotationType3D::ImproveXYPlanes => 1,
    RotationType3D::ImproveXZPlanes => 2,
});

enum_codec!(encode_fractal_type, decode_fractal_type, FractalType, "fractal type tag" {
    FractalType::None => 0,
    FractalType::FBm => 1,
    FractalType::Ridged => 2,
    FractalType::PingPong => 3,
    FractalType::DomainWarpProgressive => 4,
    FractalType::DomainWarpIndependent => 5,
});

enum_codec!(
    encode_cellular_distance_function,
    decode_cellular_distance_function,
    CellularDistanceFunction,
    "cellular distance tag" {
        CellularDistanceFunction::Euclidean => 0,
        CellularDistanceFunction::EuclideanSq => 1,
        CellularDistanceFunction::Manhattan => 2,
        CellularDistanceFunction::Hybrid => 3,
    }
);

enum_codec!(
    encode_cellular_return_type,
    decode_cellular_return_type,
    CellularReturnType,
    "cellular return tag" {
        CellularReturnType::CellValue => 0,
        CellularReturnType::Distance => 1,
        CellularReturnType::Distance2 => 2,
        CellularReturnType::Distance2Add => 3,
        CellularReturnType::Distance2Sub => 4,
        CellularReturnType::Distance2Mul => 5,
        CellularReturnType::Distance2Div => 6,
    }
);

enum_codec!(
    encode_domain_warp_type,
    decode_domain_warp_type,
    DomainWarpType,
    "domain warp tag" {
        DomainWarpType::OpenSimplex2 => 0,
        DomainWarpType::OpenSimplex2Reduced => 1,
        DomainWarpType::BasicGrid => 2,
    }
);

enum_codec!(encode_combine_op, decode_combine_op, CombineOp, "combine op tag" {
    CombineOp::Add => 0,
    CombineOp::Sub => 1,
    CombineOp::Mul => 2,
    CombineOp::Min => 3,
    CombineOp::Max => 4,
    CombineOp::Clamp => 5,
    CombineOp::Remap => 6,
    CombineOp::YGradient => 7,
});

pub(crate) struct PayloadReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> PayloadReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_exact<const N: usize>(&mut self) -> BraidResult<[u8; N]> {
        let end = self.offset + N;
        if end > self.bytes.len() {
            return Err(BraidError::InvalidSpec(
                "kernel payload truncated".to_owned(),
            ));
        }
        let mut out = [0u8; N];
        out.copy_from_slice(&self.bytes[self.offset..end]);
        self.offset = end;
        Ok(out)
    }

    fn u16(&mut self) -> BraidResult<u16> {
        Ok(u16::from_le_bytes(self.read_exact()?))
    }

    fn slot(&mut self) -> BraidResult<BufferSlot> {
        Ok(BufferSlot(self.u16()?))
    }

    fn u32(&mut self) -> BraidResult<u32> {
        Ok(u32::from_le_bytes(self.read_exact()?))
    }

    fn i32(&mut self) -> BraidResult<i32> {
        Ok(i32::from_le_bytes(self.read_exact()?))
    }

    fn f32(&mut self) -> BraidResult<f32> {
        Ok(f32::from_le_bytes(self.read_exact()?))
    }

    fn noise(&mut self) -> BraidResult<Arc<FastNoiseLite>> {
        let seed = self.i32()?;
        let frequency = self.f32()?;
        let noise_type = decode_noise_type(self.u32()?)?;
        let rotation_type = decode_rotation_type(self.u32()?)?;
        let fractal_type = decode_fractal_type(self.u32()?)?;
        let octaves = self.i32()?;
        let lacunarity = self.f32()?;
        let gain = self.f32()?;
        let weighted_strength = self.f32()?;
        let ping_pong_strength = self.f32()?;
        let cellular_distance = decode_cellular_distance_function(self.u32()?)?;
        let cellular_return = decode_cellular_return_type(self.u32()?)?;
        let cellular_jitter = self.f32()?;
        let domain_warp_type = decode_domain_warp_type(self.u32()?)?;
        let domain_warp_amp = self.f32()?;

        let mut noise = FastNoiseLite::with_seed(seed);
        noise.set_frequency(Some(frequency));
        noise.set_noise_type(Some(noise_type));
        noise.set_rotation_type_3d(Some(rotation_type));
        noise.set_fractal_type(Some(fractal_type));
        noise.set_fractal_octaves(Some(octaves));
        noise.set_fractal_lacunarity(Some(lacunarity));
        noise.set_fractal_gain(Some(gain));
        noise.set_fractal_weighted_strength(Some(weighted_strength));
        noise.set_fractal_ping_pong_strength(Some(ping_pong_strength));
        noise.set_cellular_distance_function(Some(cellular_distance));
        noise.set_cellular_return_type(Some(cellular_return));
        noise.set_cellular_jitter(Some(cellular_jitter));
        noise.set_domain_warp_type(Some(domain_warp_type));
        noise.set_domain_warp_amp(Some(domain_warp_amp));
        Ok(Arc::new(noise))
    }
}

impl KernelPayload for WarpPayload<2> {
    fn kind(&self) -> FastNoiseKernel {
        FastNoiseKernel::Warp2d
    }

    fn encode_into(&self, writer: &mut PayloadWriter<'_>) -> BraidResult<()> {
        for slot in self.source.coords {
            writer.slot(slot);
        }
        for slot in self.output.coords {
            writer.slot(slot);
        }
        writer.noise(self.noise.as_ref());
        Ok(())
    }

    fn decode_from(kind: FastNoiseKernel, reader: &mut PayloadReader<'_>) -> BraidResult<Self> {
        let _ = kind;
        Ok(Self {
            source: read_position_slots(reader)?,
            output: read_position_slots(reader)?,
            noise: reader.noise()?,
        })
    }
}

impl KernelPayload for WarpPayload<3> {
    fn kind(&self) -> FastNoiseKernel {
        FastNoiseKernel::Warp3d
    }

    fn encode_into(&self, writer: &mut PayloadWriter<'_>) -> BraidResult<()> {
        for slot in self.source.coords {
            writer.slot(slot);
        }
        for slot in self.output.coords {
            writer.slot(slot);
        }
        writer.noise(self.noise.as_ref());
        Ok(())
    }

    fn decode_from(kind: FastNoiseKernel, reader: &mut PayloadReader<'_>) -> BraidResult<Self> {
        let _ = kind;
        Ok(Self {
            source: read_position_slots(reader)?,
            output: read_position_slots(reader)?,
            noise: reader.noise()?,
        })
    }
}

impl KernelPayload for SamplePayload<2> {
    fn kind(&self) -> FastNoiseKernel {
        FastNoiseKernel::Sample2d
    }

    fn encode_into(&self, writer: &mut PayloadWriter<'_>) -> BraidResult<()> {
        for slot in self.source.coords {
            writer.slot(slot);
        }
        writer.slot(self.output);
        writer.noise(self.noise.as_ref());
        Ok(())
    }

    fn decode_from(kind: FastNoiseKernel, reader: &mut PayloadReader<'_>) -> BraidResult<Self> {
        let _ = kind;
        Ok(Self {
            source: read_position_slots(reader)?,
            output: reader.slot()?,
            noise: reader.noise()?,
        })
    }
}

impl KernelPayload for SamplePayload<3> {
    fn kind(&self) -> FastNoiseKernel {
        FastNoiseKernel::Sample3d
    }

    fn encode_into(&self, writer: &mut PayloadWriter<'_>) -> BraidResult<()> {
        for slot in self.source.coords {
            writer.slot(slot);
        }
        writer.slot(self.output);
        writer.noise(self.noise.as_ref());
        Ok(())
    }

    fn decode_from(kind: FastNoiseKernel, reader: &mut PayloadReader<'_>) -> BraidResult<Self> {
        let _ = kind;
        Ok(Self {
            source: read_position_slots(reader)?,
            output: reader.slot()?,
            noise: reader.noise()?,
        })
    }
}

fn read_position_slots<const N: usize>(
    reader: &mut PayloadReader<'_>,
) -> BraidResult<PositionSlots<N>> {
    let mut coords = [BufferSlot(0); N];
    for slot in &mut coords {
        *slot = reader.slot()?;
    }
    Ok(PositionSlots { coords })
}

impl KernelPayload for CombinePayload {
    fn kind(&self) -> FastNoiseKernel {
        FastNoiseKernel::Combine
    }

    fn encode_into(&self, writer: &mut PayloadWriter<'_>) -> BraidResult<()> {
        writer.u32(encode_combine_op(self.op));
        writer.u32(self.inputs.len() as u32);
        for slot in &self.inputs {
            writer.slot(*slot);
        }
        writer.slot(self.output);
        writer.u32(self.params.len() as u32);
        for value in &self.params {
            writer.f32(*value);
        }
        Ok(())
    }

    fn decode_from(kind: FastNoiseKernel, reader: &mut PayloadReader<'_>) -> BraidResult<Self> {
        let _ = kind;
        let op = decode_combine_op(reader.u32()?)?;
        let input_count = reader.u32()? as usize;
        let mut inputs = Vec::with_capacity(input_count);
        for _ in 0..input_count {
            inputs.push(reader.slot()?);
        }
        let output = reader.slot()?;
        let param_count = reader.u32()? as usize;
        let mut params = Vec::with_capacity(param_count);
        for _ in 0..param_count {
            params.push(reader.f32()?);
        }
        Ok(Self {
            op,
            inputs,
            output,
            params,
        })
    }
}
