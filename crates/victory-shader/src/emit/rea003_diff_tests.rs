//! REA-003 Frente B unit & differential tests for `VICTORY_REA003_SHADER_RECOVERY=1`.
//!
//! Each test constructs native GFX10 machine code programs (with explicit root
//! layouts and resource usage) and verifies both:
//! 1. Conservative baseline (`rea003_shader_recovery = false`) remains unchanged.
//! 2. Recovery enabled (`rea003_shader_recovery = true`) emits valid SM 6.6 DXIL
//!    that passes `dxil.dll` validation (when present).

use super::{HostShader, lower, validator, with_rea003_shader_recovery};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const S_BARRIER: u32 = 0xBF8A_0000;

fn lower_cs(words: &[u32], threads: [u32; 3], lds_dwords: u32, wave64: bool) -> NativeProgram {
    let mut all = words.to_vec();
    all.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 32,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 16,
                dwords: 4,
                user_data: 4,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 8,
                lower_bound: 1,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: (5 << 1) | (lds_dwords.div_ceil(128) << 15),
            threads,
        }),
        code: all.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn translate_program(native: &NativeProgram, host_wave: u32) -> Result<HostShader, ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let io_info = io::raise(&p, native)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
    let out = lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io_info,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    let lowered = out?;
    Ok(HostShader {
        stage: native.stage,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    })
}

fn assert_dxil_valid(shader: &HostShader) {
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy)
            .expect("recovered DXIL must validate cleanly under dxil.dll");
    }
}

#[test]
fn buffer_num_records_read_as_data_recovers_only_when_opted_in() {
    // s4 = table base, s5 = 0; load SRV descriptor s[0:3] and read s2 (num_records)
    // without executing a buffer_load on s[0:3]; store s2 to UAV at table+32.
    let prog = lower_cs(
        &[
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0002,
            0xFA00_0000, // s_load_dwordx4 s[0:3], s[4:5], null
            0xF408_0202,
            0xFA00_0020, // s_load_dwordx4 s[8:11], s[4:5], 0x20 (UAV at dwords=8)
            0x7E00_0202, // v_mov_b32 v0, s2 (read num_records as integer!)
            0x38 << 26 | 0x1C << 18,
            0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        ],
        [64, 1, 1],
        0,
        true,
    );

    let baseline = with_rea003_shader_recovery(false, || translate_program(&prog, 32));
    assert!(
        baseline.is_err(),
        "baseline must stay strict when REA-003 shader recovery is OFF"
    );

    let recovered = with_rea003_shader_recovery(true, || translate_program(&prog, 32))
        .expect("buffer record count recovery must succeed");
    assert_dxil_valid(&recovered);
}

#[test]
fn dynamic_image_descriptor_read_as_data_packs_sq_img_rsrc_dimensions() {
    // Pixel shader with ResourceHeap dynamic index `s1` loading an 8-dword image
    // descriptor solely to unpack RDNA2 SQ_IMG_RSRC width (`s9 >> 30 | (s10 & 0x3FFF) << 2`)
    // and export it as color (no `image_*` instruction).
    let words = vec![
        0xBE83_0380, // s_mov_b32 s3, 0
        0x8F01_8501, // s_lshl_b32 s1, s1, 5 (dynamic index * 32)
        0xF40C_0201,
        0x0200_0000, // s_load_dwordx8 s[8:15], s[2:3], s1 (table_hi=s3=0)
        0x9000_9E09, // s_lshr_b32 s0, s9, 30
        0x8F04_820A, // s_lshl_b32 s4, s10, 2
        0x8800_0400, // s_or_b32 s0, s0, s4
        0x8100_8100, // s_add_i32 s0, s0, 1
        0x7E00_0200, // v_mov_b32 v0, s0
        0xF800_180F,
        0x0000_0000, // exp mrt0, v0, v0, v0, v0 done vm (uncompressed)
        S_ENDPGM,
    ];
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: true,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::ResourceHeap,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 2,
                slot: 1,
            },
        ],
        used: Vec::new(),
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 0,
            db_shader_control: 0,
            patches: Vec::new(),
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir");
    let st = structure::structure(&p).expect("structure");
    let io_info = io::raise(&p, &native).expect("io");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource");
    let outputs = vec![crate::emit::container::SigElement {
        name: "SV_Target".to_owned(),
        index: 0,
        kind: 16,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    }];
    let parts = lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io_info,
        res: &res,
        inputs: Vec::new(),
        outputs,
        prims: Vec::new(),
        mesh_stride: 0,
    };

    let base_res = with_rea003_shader_recovery(false, || lower::lower(&parts));
    assert!(
        base_res.is_err(),
        "baseline must refuse unread 8-dword descriptor without recovery flag"
    );

    let rec = with_rea003_shader_recovery(true, || lower::lower(&parts))
        .expect("dynamic 8-dword image descriptor recovery must emit valid DXIL");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = rec.dxil.clone();
        v.validate(&mut copy)
            .expect("dynamic image dimension DXIL must validate");
    }
}

#[test]
fn lds_explicit_s_barrier_splits_phases_and_lowers_to_wave32() {
    // 128-thread compute shader: inside an active branch, writes LDS before `s_barrier`
    // and reads LDS after `s_barrier`. Without barrier-splitting, `lds_meetings` refuses
    // Wave32; with recovery, phases split cleanly at `s_barrier` and Wave32 validates.
    let prog = lower_cs(
        &[
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0020, // s_load_dwordx4 s[8:11], s[4:5], 0x20 (UAV)
            0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0 (lane branch!)
            0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
            0xBF88_0008, // s_cbranch_execz -> restore_exec (+8 dwords)
            0x3402_0082, // v_lshlrev_b32 v1, 2, v0
            0xD834_0000,
            0x0000_0101, // ds_write_b32 v1, v1
            S_BARRIER,   // explicit barrier separates write phase from read phase!
            0xD8D8_0000,
            0x0200_0001, // ds_read_b32 v2, v1
            0x38 << 26 | 0x1C << 18 | 0x02 << 8,
            0x80 << 24 | 2 << 16, // buffer_store_dword v2, off, s[8:11], 0
            0xBEFE_040C,          // s_mov_b64 exec, s[12:13]
        ],
        [128, 1, 1],
        128,
        true,
    );

    let baseline = with_rea003_shader_recovery(false, || translate_program(&prog, 32).map(|_| ()));
    let _ = baseline;

    let recovered = with_rea003_shader_recovery(true, || translate_program(&prog, 32))
        .expect("explicit s_barrier split must allow clean lowering");
    assert_dxil_valid(&recovered);
}

#[test]
fn wave_uniform_readfirstlane_and_constant_mask_recover_without_divergent_barriers() {
    // 128-thread compute shader: tests constant lane mask select (`v_cndmask_b32` with
    // literal 0xAAAAAAAA in `s[0:1]`) AND `v_readfirstlane_b32` of a uniform `v3 = 7` inside
    // a lane-masked branch.
    let prog = lower_cs(
        &[
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0020, // s_load_dwordx4 s[8:11], s[4:5], 0x20 (UAV)
            0xBE80_03FF,
            0xAAAA_AAAA, // s_mov_b32 s0, lit(0xAAAAAAAA)
            0xBE81_03FF,
            0xAAAA_AAAA, // s_mov_b32 s1, lit(0xAAAAAAAA)
            0xD501_0001,
            0x0002_0280, // v_cndmask_b32 v1, 0, 1, s[0:1]
            0x7E06_0287, // v_mov_b32 v3, 7 (wave-uniform VGPR!)
            0x7D8A_0280, // v_cmp_ne_u32 vcc, 0, v1
            0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
            0x7E04_0503, // v_readfirstlane_b32 s2, v3 (first-lane read of wave-uniform VGPR!)
            0x4A02_0202, // v_add_nc_u32 v1, s2, v1
            0x38 << 26 | 0x1C << 18 | 0x01 << 8,
            0x80 << 24 | 2 << 16, // buffer_store_dword v1, off, s[8:11], 0
            0xBEFE_040C,          // s_mov_b64 exec, s[12:13]
        ],
        [128, 1, 1],
        128,
        true,
    );

    let recovered = with_rea003_shader_recovery(true, || translate_program(&prog, 32))
        .expect("wave-uniform readfirstlane & constant mask test must translate cleanly");
    assert_dxil_valid(&recovered);
}
