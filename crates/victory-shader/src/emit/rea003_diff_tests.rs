//! REA-003 Frente B unit & differential tests for `VICTORY_REA003_SHADER_RECOVERY=1`.
//!
//! Each test constructs native GFX10 machine code programs (with explicit root
//! layouts and resource usage) and verifies both:
//! 1. Conservative baseline (`rea003_shader_recovery = false`) remains unchanged.
//! 2. Recovery enabled (`rea003_shader_recovery = true`) emits valid SM 6.6 DXIL
//!    that passes `dxil.dll` validation (when present).

use super::{
    HostShader, host, lower, validator, with_rea003_recovery_lists, with_rea003_shader_recovery,
    with_shader_opt,
};
use crate::emit::bitcode::opt::{ALL, ALL_REA003};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};
use lower::rea003_recovery;

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
fn image_descriptor_recovery_rejects_resource_heap_and_requires_typed_table_entry() {
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
    let make_ps = |root_ty: RootType, used: Vec<UsedResource>| NativeProgram {
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
                root: root_ty,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 2,
                slot: 1,
            },
        ],
        used,
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 0,
            db_shader_control: 0,
            patches: Vec::new(),
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let lower_ps = |native: &NativeProgram| {
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir");
        let st = structure::structure(&p).expect("structure");
        let io_info = io::raise(&p, native).expect("io");
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
            native,
            p: &p,
            st: &st,
            io: &io_info,
            res: &res,
            inputs: Vec::new(),
            outputs,
            prims: Vec::new(),
            mesh_stride: 0,
        };
        lower::lower(&parts)
    };

    // Untyped ResourceHeap with no `used` entry MUST be rejected even with `img_meta` allowed.
    let untyped_heap = make_ps(RootType::ResourceHeap, Vec::new());
    with_rea003_shader_recovery(true, || {
        with_rea003_recovery_lists(Some("img_meta"), None, || {
            assert!(
                lower_ps(&untyped_heap).is_err(),
                "untyped ResourceHeap image descriptor recovery must be rejected"
            );
        });
    });

    // Typed Table 8-dword SRV: rejected by default allowlist, accepted ONLY when `img_meta` opted in.
    let typed_table = NativeProgram {
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
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 2,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 2,
            slot: 1,
        }],
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 0,
            db_shader_control: 0,
            patches: Vec::new(),
        },
        code: [
            0xBE83_0380, // s_mov_b32 s3, 0
            0xF40C_0201,
            0xFA00_0000, // s_load_dwordx8 s[8:15], s[2:3], null
            0x9000_9E09, // s_lshr_b32 s0, s9, 30
            0x8F04_820A, // s_lshl_b32 s4, s10, 2
            0x8800_0400, // s_or_b32 s0, s0, s4
            0x8100_8100, // s_add_i32 s0, s0, 1
            0x7E00_0200, // v_mov_b32 v0, s0
            0xF800_180F,
            0x0000_0000, // exp mrt0
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect(),
    };
    with_rea003_shader_recovery(true, || {
        assert!(
            lower_ps(&typed_table).is_err(),
            "img_meta must stay OFF under conservative default allowlist"
        );
        with_rea003_recovery_lists(Some("img_meta"), None, || {
            let rec = lower_ps(&typed_table)
                .expect("typed Table 8-dword SRV image recovery must succeed when opted in");
            if let Ok(mut v) =
                validator::Validator::load(validator::Validator::default_path().as_deref())
            {
                let mut copy = rec.dxil.clone();
                v.validate(&mut copy)
                    .expect("typed image dimension DXIL must validate");
            }
        });
    });
}

#[test]
fn lds_explicit_s_barrier_rejects_divergent_barrier_and_allows_uniform_barrier() {
    // Case 1: Divergent `s_barrier` inside a lane-masked branch MUST be rejected.
    let divergent_barrier = lower_cs(
        &[
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0020, // s_load_dwordx4 s[8:11], s[4:5], 0x20 (UAV)
            0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0 (lane branch -> !all[b])
            0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
            0xBF88_0008, // s_cbranch_execz +8 dwords
            0x3402_0082, // v_lshlrev_b32 v1, 2, v0
            0xD834_0000,
            0x0000_0101, // ds_write_b32 v1, v1
            S_BARRIER,   // DIVERGENT barrier!
            0xD8D8_0000,
            0x0200_0001, // ds_read_b32 v2, v1
            0x38 << 26 | 0x1C << 18 | 0x02 << 8,
            0x80 << 24 | 2 << 16, // buffer_store_dword
            0xBEFE_040C,          // s_mov_b64 exec, s[12:13]
        ],
        [64, 1, 1],
        128,
        true,
    );
    let p_div = ir::build(&divergent_barrier.code, true, 32).unwrap();
    let st_div = structure::structure(&p_div).unwrap();
    let all_div: Vec<bool> = (0..st_div.exits.len())
        .map(|b| {
            p_div.blocks[b]
                .insts
                .iter()
                .all(|i| i.name() != "s_barrier")
        })
        .collect();
    assert!(
        rea003_recovery::lds_meetings_with_barriers(&p_div, &st_div, &all_div).is_none(),
        "divergent s_barrier inside !all[b] must NEVER be admitted"
    );

    // Case 2: Atomic LDS RMW (`ds_add_u32`) + read in one divergent region must return None.
    let atomic_div = lower_cs(
        &[
            0xBE85_0380,
            0xF408_0202,
            0xFA00_0020,
            0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
            0xBE8C_246A, // s_and_saveexec_b64
            0xBF88_0007,
            0x3402_0082, // v_lshlrev_b32 v1, 2, v0
            0xD800_0000,
            0x0000_0101, // ds_add_u32 v1, v1 (atomic RMW -> both read and write!)
            0xD8D8_0000,
            0x0200_0001, // ds_read_b32 v2, v1
            0x38 << 26 | 0x1C << 18 | 0x02 << 8,
            0x80 << 24 | 2 << 16,
            0xBEFE_040C,
        ],
        [64, 1, 1],
        128,
        true,
    );
    let p_atom = ir::build(&atomic_div.code, true, 32).unwrap();
    let st_atom = structure::structure(&p_atom).unwrap();
    let all_atom: Vec<bool> = (0..st_atom.exits.len())
        .map(|b| {
            !p_atom.blocks[b]
                .insts
                .iter()
                .any(|i| i.name() == "ds_add_u32")
        })
        .collect();
    assert!(
        rea003_recovery::lds_meetings_with_barriers(&p_atom, &st_atom, &all_atom).is_none(),
        "atomic LDS RMW must be classified as read+write and refuse divergent barrier-less lowering"
    );

    // Case 3: Uniform `s_barrier` in `all[b]` between separate `!all[b]` write and read blocks.
    let uniform_barrier = lower_cs(
        &[
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0020, // s_load_dwordx4 s[8:11], s[4:5], 0x20
            0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
            0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
            0xBF88_0003, // s_cbranch_execz +3 dwords
            0x3402_0082, // v_lshlrev_b32 v1, 2, v0
            0xD834_0000,
            0x0000_0101, // ds_write_b32 v1, v1
            0xBEFE_040C, // s_mov_b64 exec, s[12:13] (reconverged -> all[b] == true!)
            S_BARRIER,   // uniform barrier
            0x7D88_0000, // second divergent branch for read
            0xBE8C_246A,
            0xBF88_0005, // s_cbranch_execz +5
            0x3402_0082,
            0xD8D8_0000,
            0x0200_0001, // ds_read_b32 v2, v1
            0xE070_1000,
            0x8002_0201, // buffer_store_dword v2, v1, s[8:11], 0 offen
            0xBEFE_040C,
        ],
        [64, 1, 1],
        128,
        true,
    );
    with_rea003_shader_recovery(true, || {
        with_rea003_recovery_lists(Some("lds_barrier"), None, || {
            let recovered = translate_program(&uniform_barrier, 32)
                .expect("uniform s_barrier between separate LDS phases must lower to Wave32");
            assert_dxil_valid(&recovered);
        });
    });
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

#[cfg(windows)]
fn lower_cs_gid(words: &[u32], lds_dwords: u32, wave64: bool) -> NativeProgram {
    // Prefix: save GroupId(0) (delivered in s5 after 5 user SGPRs) into s7,
    // zero s5 for 64-bit table base s[4:5], load UAV s[8:11] at +0x20,
    // compute gid = (s7 << 6) + v0 in v7 and byte offset (gid << 2) in v8.
    let mut full = vec![
        0xBE87_0305, // s_mov_b32 s7, s5
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0020, // s_load_dwordx4 s[8:11], s[4:5], 0x20 (UAV)
        0x8F07_8607, // s_lshl_b32 s7, s7, 6
        0x4A0E_0007, // v_add_nc_u32 v7, s7, v0
        0x3410_0E82, // v_lshlrev_b32 v8, 2, v7
    ];
    full.extend_from_slice(words);
    full.push(S_ENDPGM);
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
            rsrc2: (5 << 1) | (1 << 7) | (lds_dwords.div_ceil(128) << 15),
            threads: [64, 1, 1],
        }),
        code: full.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

#[cfg(windows)]
fn dispatch_cs_on_host(
    native: &NativeProgram,
    host_wave: u32,
    k: [u32; 4],
    srv_words: &[u32],
    srv_stride: u32,
    uav_dwords: u32,
    groups: [u32; 3],
) -> Option<(HostShader, Vec<u32>)> {
    let cs = translate_program(native, host_wave).expect("shader must translate");
    assert_dxil_valid(&cs);
    let Ok(gpu) = host::Host::new(false) else {
        eprintln!("no SM 6.6 device; GPU dispatch skipped");
        return None;
    };
    let (root, pso) = gpu
        .compute_pipeline(&cs)
        .expect("D3D12 compute PSO creation must succeed");
    let mut constants = vec![0u32; cs.user_data_dwords().max(5) as usize];
    for u in &cs.user_data {
        if u.root == RootType::Constants && (u.dword as usize) < k.len() {
            constants[u.index as usize] = k[u.dword as usize];
        } else if u.root == RootType::Table {
            constants[u.index as usize] = 0;
        }
    }
    let dummy = [0u32; 4];
    let srv = if srv_words.is_empty() {
        &dummy[..]
    } else {
        srv_words
    };
    let stride = if srv_stride > 0 {
        host::Stride::Structured(srv_stride)
    } else {
        host::Stride::Raw
    };
    let views = [
        host::View::Buffer { data: srv, stride },
        host::View::RawUav(uav_dwords),
    ];
    let out = gpu
        .dispatch(
            &root,
            &pso,
            &host::Dispatch {
                roots: vec![(0, host::Root::Constants(constants))],
                views: &views,
                samplers: &[],
                groups,
            },
        )
        .expect("D3D12 compute dispatch must succeed");
    Some((cs, out.into_iter().next().unwrap()))
}

#[cfg(windows)]
#[test]
fn gpu_exec_divergent_control_flow_and_reconvergence_256_threads() {
    let prog = lower_cs_gid(
        &[
            0x7E02_0307,      // v_mov_b32 v1, v7
            0x4A02_028A,      // v_add_nc_u32 v1, 10, v1
            0x7D88_0080 | 16, // v_cmp_gt_u32 vcc, 16, v0 (lanes 0..15)
            0xBE8C_246A,      // s_and_saveexec_b64 s[12:13], vcc
            0xBF88_0001,      // s_cbranch_execz +1
            0x4A02_02B2,      // v_add_nc_u32 v1, 50, v1
            0xBEFE_040C,      // s_mov_b64 exec, s[12:13]
            0x4A02_0281,      // v_add_nc_u32 v1, 1, v1 (reconverged)
            0xE070_1000,
            0x8002_0108, // buffer_store_dword v1, v8, s[8:11], 0 offen
        ],
        0,
        true,
    );
    with_rea003_shader_recovery(true, || {
        let Some((_, got)) = dispatch_cs_on_host(&prog, 32, [0; 4], &[], 0, 256, [4, 1, 1]) else {
            return;
        };
        assert_eq!(got.len(), 256);
        for (gid, &val) in got.iter().enumerate() {
            let local = gid % 64;
            let expected = (gid as u32) + 11 + if local < 16 { 50 } else { 0 };
            assert_eq!(val, expected, "gid={gid} local={local}");
        }
    });
}

#[cfg(windows)]
#[test]
fn gpu_exec_const_mask_cndmask_covers_all_64_lanes_across_4_groups() {
    // Asymmetric 64-bit mask: lo = 0x5555_5555 (even lanes in 0..31),
    // hi = 0xAAAA_AAAA (odd lanes in 32..63). Proves lanes 32..63 read s1, not s0!
    let prog = lower_cs_gid(
        &[
            0x7E02_028A, // v_mov_b32 v1, 10
            0x7E04_0294, // v_mov_b32 v2, 20
            0xD501_0001,
            0x0002_0501, // v_cndmask_b32 v1, v1, v2, s[0:1]
            0xE070_1000,
            0x8002_0108, // buffer_store_dword v1, v8, s[8:11], 0 offen
        ],
        0,
        true,
    );
    let mask_lo = 0x5555_5555u32;
    let mask_hi = 0xAAAA_AAAAu32;
    let mask64 = u64::from(mask_lo) | (u64::from(mask_hi) << 32);
    with_rea003_shader_recovery(true, || {
        let Some((_, got)) =
            dispatch_cs_on_host(&prog, 32, [mask_lo, mask_hi, 0, 0], &[], 0, 256, [4, 1, 1])
        else {
            return;
        };
        assert_eq!(got.len(), 256);
        for (gid, &val) in got.iter().enumerate() {
            let local = gid % 64;
            let bit = (mask64 >> local) & 1;
            let expected = if bit != 0 { 20 } else { 10 };
            assert_eq!(val, expected, "gid={gid} local={local} bit={bit}");
        }
    });
}

#[cfg(windows)]
#[test]
fn gpu_exec_divergent_phi_readfirstlane_256_threads() {
    // Divergent branch sets v3 differently for lanes < 32 vs lanes >= 32 (yielding a
    // divergent Phi); subsequent `v_readfirstlane_b32 s2, v3` MUST detect non-uniformity
    // (`is_wave_uniform_val == false`) and exchange halves on Wave32 so ALL 64 threads
    // in every group observe lane 0's value (`group * 64 + 51`), not half-local lane 32!
    let prog = lower_cs_gid(
        &[
            0x7E08_028A, // v_mov_b32 v4, 10
            0x7E0A_0294, // v_mov_b32 v5, 20
            0xD501_0004,
            0x0002_0B04, // v_cndmask_b32 v4, v4, v5, s[0:1] (forces 64-lane exchange halves!)
            0x4A06_0E81, // v_add_nc_u32 v3, 1, v7 (v3 = gid + 1)
            0x7D88_00A0, // v_cmp_gt_u32 vcc, 32, v0 (lanes 0..31)
            0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
            0xBF88_0001, // s_cbranch_execz +1
            0x4A06_06B2, // v_add_nc_u32 v3, 50, v3 (lanes 0..31 -> gid + 51!)
            0xBEFE_040C, // s_mov_b64 exec, s[12:13]
            0x7E04_0503, // v_readfirstlane_b32 s2, v3 (Phi!)
            0x7E02_0202, // v_mov_b32 v1, s2
            0xE070_1000,
            0x8002_0108, // buffer_store_dword v1, v8, s[8:11], 0 offen
        ],
        0,
        true,
    );
    let p = ir::build(&prog.code, true, 32).unwrap();
    let st = structure::structure(&p).unwrap();
    let v3_phi = p
        .blocks
        .iter()
        .find_map(|b| {
            b.insts
                .iter()
                .find(|i| i.name() == "v_readfirstlane_b32")
                .and_then(|i| i.read(crate::ir::Reg::V(3)))
        })
        .unwrap();
    assert!(
        !rea003_recovery::is_wave_uniform_val(&p, &st, None, v3_phi),
        "divergent Phi feeding v_readfirstlane_b32 must NOT be marked wave-uniform"
    );
    // Pure-constant divergent Phi (`v3 = 1`, `if (v0 < 32) v3 = 50`, join Phi):
    // incoming operands are pure constants, so ONLY checking `!all[pred]` catches it!
    let const_phi_prog = lower_cs_gid(
        &[
            0x7E08_028A, // v_mov_b32 v4, 10
            0x7E0A_0294, // v_mov_b32 v5, 20
            0xD501_0004,
            0x0002_0B04, // v_cndmask_b32 v4, v4, v5, s[0:1] (forces 64-lane exchange halves!)
            0x7E06_0281, // v_mov_b32 v3, 1
            0x7D88_00A0, // v_cmp_gt_u32 vcc, 32, v0 (lanes 0..31)
            0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
            0xBF88_0001, // s_cbranch_execz +1
            0x7E06_02B2, // v_mov_b32 v3, 50 (pure constant inside divergent branch!)
            0xBEFE_040C, // s_mov_b64 exec, s[12:13]
            0x7E04_0503, // v_readfirstlane_b32 s2, v3 (Phi!)
            0x7E02_0202, // v_mov_b32 v1, s2
            0xE070_1000,
            0x8002_0108, // buffer_store_dword v1, v8, s[8:11], 0 offen
        ],
        0,
        true,
    );
    let p_cp = ir::build(&const_phi_prog.code, true, 32).unwrap();
    let st_cp = structure::structure(&p_cp).unwrap();
    let all_cp: Vec<bool> = (0..st_cp.exits.len())
        .map(|b| {
            !(p_cp.blocks[b].insts.len() == 1 && p_cp.blocks[b].insts[0].name() == "v_mov_b32")
        })
        .collect();
    let v3_cp = p_cp
        .blocks
        .iter()
        .find_map(|b| {
            b.insts
                .iter()
                .find(|i| i.name() == "v_readfirstlane_b32")
                .and_then(|i| i.read(crate::ir::Reg::V(3)))
        })
        .unwrap();
    assert!(
        !rea003_recovery::is_wave_uniform_val(&p_cp, &st_cp, Some(&all_cp), v3_cp),
        "pure-constant divergent Phi must NOT be marked wave-uniform when pred has !all"
    );
    with_rea003_shader_recovery(true, || {
        let Some((_, got)) = dispatch_cs_on_host(&prog, 32, [0; 4], &[], 0, 256, [4, 1, 1]) else {
            return;
        };
        assert_eq!(got.len(), 256);
        for (gid, &val) in got.iter().enumerate() {
            let group_base = (gid / 64) * 64;
            let expected = (group_base as u32) + 51;
            assert_eq!(val, expected, "gid={gid} must observe lane 0 value");
        }
        let Some((_, got_cp)) =
            dispatch_cs_on_host(&const_phi_prog, 32, [0; 4], &[], 0, 256, [4, 1, 1])
        else {
            return;
        };
        assert_eq!(got_cp.len(), 256);
        assert!(
            got_cp.iter().all(|&v| v == 50),
            "pure-constant divergent Phi must exchange across Wave32 halves so all 256 threads observe 50"
        );
    });
}

#[cfg(windows)]
#[test]
fn gpu_exec_uniform_barrier_lds_exchange_256_threads() {
    // Branch 1 writes LDS[(v0 << 2)] = gid + 10 for lanes 0..63; reconverged uniform `s_barrier`;
    // Branch 2 reads LDS[((v0 ^ 32) << 2)] across the OTHER Wave32 half!
    let prog = lower_cs_gid(
        &[
            0x3402_0082, // v_lshlrev_b32 v1, 2, v0 (v0 * 4)
            0x4A04_0E8A, // v_add_nc_u32 v2, 10, v7 (gid + 10)
            0xD834_0000,
            0x0000_0201, // ds_write_b32 v1, v2
            S_BARRIER,   // uniform barrier synchronizing both Wave32 halves
            0x3A08_00A0, // v_xor_b32 v4, 32, v0 (partner in opposite half!)
            0x3408_0882, // v_lshlrev_b32 v4, 2, v4
            0xD8D8_0000,
            0x0100_0004, // ds_read_b32 v1, v4
            0xE070_1000,
            0x8002_0108, // buffer_store_dword v1, v8, s[8:11], 0 offen
        ],
        128,
        true,
    );
    with_rea003_shader_recovery(true, || {
        with_rea003_recovery_lists(Some("lds_barrier"), None, || {
            let Some((_, got)) = dispatch_cs_on_host(&prog, 32, [0; 4], &[], 0, 256, [4, 1, 1])
            else {
                return;
            };
            assert_eq!(got.len(), 256);
            for (gid, &val) in got.iter().enumerate() {
                let partner_gid = gid ^ 32;
                let expected = (partner_gid as u32) + 10;
                assert_eq!(val, expected, "gid={gid} partner={partner_gid}");
            }
        });
    });
}

#[cfg(windows)]
#[test]
fn gpu_exec_buffer_num_records_metadata_recovery() {
    let prog = lower_cs_gid(
        &[
            0xF408_0002,
            0xFA00_0000, // s_load_dwordx4 s[0:3], s[4:5], null (SRV at table+0)
            0x7E02_0202, // v_mov_b32 v1, s2 (read num_records without buffer_load!)
            0xE070_1000,
            0x8002_0108, // buffer_store_dword v1, v8, s[8:11], 0 offen
        ],
        0,
        true,
    );
    let srv_words = vec![0xABCD_1234u32; 256]; // 1024 bytes / stride 16 = 64 records!
    with_rea003_shader_recovery(true, || {
        let Some((_, got)) = dispatch_cs_on_host(&prog, 32, [0; 4], &srv_words, 16, 256, [4, 1, 1])
        else {
            return;
        };
        assert_eq!(got.len(), 256);
        assert!(
            got.iter().all(|&v| v == 64),
            "all 256 threads must read exact num_records=64 from getDimensions; got {:?}",
            &got[..8]
        );
    });
}

#[cfg(windows)]
#[test]
fn gpu_exec_shader_opt_bitfield_licm_smaller_and_bit_exact() {
    // Shader with invariant arithmetic + nested redundant masking inside a 4-iteration loop.
    let prog = lower_cs_gid(
        &[
            0xBE8E_0380, // s_mov_b32 s14, 0
            0x7E02_0280, // v_mov_b32 v1, 0
            // loop body: 6 words before s_cbranch_scc1
            0x7E04_0200, // v_mov_b32 v2, s0
            0x3604_049F, // v_and_b32 v2, 31, v2
            0x3604_048F, // v_and_b32 v2, 15, v2
            0x4A04_0487, // v_add_nc_u32 v2, 7, v2
            0x4A02_0302, // v_add_nc_u32 v1, v2, v1
            0x810E_810E, // s_add_i32 s14, s14, 1
            0xBF0A_840E, // s_cmp_lt_u32 s14, 4
            0xBF85_FFF8, // s_cbranch_scc1 -8
            0x4A02_0307, // v_add_nc_u32 v1, v7, v1
            0xE070_1000,
            0x8002_0108, // buffer_store_dword v1, v8, s[8:11], 0 offen
        ],
        0,
        true,
    );
    let Some((base_cs, base_out)) = with_shader_opt(ALL, || {
        dispatch_cs_on_host(&prog, 32, [0x2A, 0, 0, 0], &[], 0, 256, [4, 1, 1])
    }) else {
        return;
    };
    let Some((opt_cs, opt_out)) = with_shader_opt(ALL_REA003, || {
        dispatch_cs_on_host(&prog, 32, [0x2A, 0, 0, 0], &[], 0, 256, [4, 1, 1])
    }) else {
        return;
    };
    assert_eq!(
        base_out, opt_out,
        "BITFIELD+LICM must preserve bit-exact GPU output"
    );
    let inv = ((0x2A & 31) & 15) + 7; // (42 & 15) + 7 = 10 + 7 = 17; 4 * 17 = 68
    for (gid, &val) in opt_out.iter().enumerate() {
        assert_eq!(val, 4 * inv + gid as u32, "gid={gid}");
    }
    assert!(
        opt_cs.dxil.len() <= base_cs.dxil.len(),
        "ALL_REA003 DXIL ({}) should not exceed ALL ({})",
        opt_cs.dxil.len(),
        base_cs.dxil.len()
    );
}
