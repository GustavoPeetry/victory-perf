//! CPU Möller–Trumbore reference intersector and RTX 2060 SUPER DXR 1.1+ harness tests.

use crate::dxr_rayquery::{RayHitOutput, RayInput};

/// Transforms a 3D point by a row-major 3x4 affine instance matrix.
pub fn transform_point_3x4(m: &[f32; 12], p: [f32; 3]) -> [f32; 3] {
    [
        m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
        m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
        m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
    ]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// CPU scene mesh slice paired with instance transform, ID, and inclusion mask.
pub struct CpuInstance<'a> {
    pub triangles: &'a [[[f32; 3]; 3]],
    pub transform_3x4: [f32; 12],
    pub instance_id: u32,
    pub mask: u8,
}

/// Möller–Trumbore closest-hit reference matching DXR `TraceRayInline` closest hit & barycentrics.
pub fn trace_cpu_reference(instances: &[CpuInstance<'_>], ray: &RayInput) -> RayHitOutput {
    let mut best = RayHitOutput {
        hit: 0,
        t: ray.tmax,
        bary_u: 0.0,
        bary_v: 0.0,
        primitive_index: 0,
        instance_id: 0,
        geometry_index: 0,
        front_face: 0,
    };
    let ray_mask = (ray.instance_mask & 0xFF) as u8;

    for inst in instances {
        if (inst.mask & ray_mask) == 0 {
            continue;
        }
        for (prim_idx, tri) in inst.triangles.iter().enumerate() {
            let v0 = transform_point_3x4(&inst.transform_3x4, tri[0]);
            let v1 = transform_point_3x4(&inst.transform_3x4, tri[1]);
            let v2 = transform_point_3x4(&inst.transform_3x4, tri[2]);

            let e1 = sub(v1, v0);
            let e2 = sub(v2, v0);
            let pvec = cross(ray.dir, e2);
            let det = dot(e1, pvec);
            if det.abs() < 1e-12 {
                continue;
            }
            let inv_det = 1.0 / det;
            let tvec = sub(ray.origin, v0);
            let u = dot(tvec, pvec) * inv_det;
            if !(0.0..=1.0).contains(&u) {
                continue;
            }
            let qvec = cross(tvec, e1);
            let v = dot(ray.dir, qvec) * inv_det;
            if v < 0.0 || u + v > 1.0 {
                continue;
            }
            let t = dot(e2, qvec) * inv_det;
            if t >= ray.tmin && t <= best.t {
                best.hit = 1;
                best.t = t;
                best.bary_u = u;
                best.bary_v = v;
                best.primitive_index = prim_idx as u32;
                best.instance_id = inst.instance_id;
                best.geometry_index = 0;
                // Default DXR winding treats clockwise triangles in LH space as front-facing.
                best.front_face = u32::from(det > 0.0);
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dxr_backend::{DxrBackend, DxrInstance, DxrTriangleMeshDesc, INDEX_FORMAT_U32};
    use crate::dxr_rayquery::{DxrRayQueryPipeline, semantic_mapping};

    #[test]
    fn dxr_flag_defaults_off_and_semantic_mapping_covers_bvh_outputs() {
        assert!(!crate::dxr_backend::dxr_enabled());
        let map = semantic_mapping();
        assert!(map.len() >= 4);
        assert!(
            map.iter()
                .any(|m| m.rdna2_component.contains("barycentrics"))
        );
        assert!(
            map.iter()
                .any(|m| m.dxr_rayquery_equivalent.contains("TraceRayInline"))
        );
    }

    #[test]
    fn dxr_rayquery_matches_cpu_reference_on_host_gpu() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let tier = crate::dxr_proto::probe_tier(&host.device).unwrap_or(0);
        if tier < 11 {
            return;
        }
        let mut backend = DxrBackend::new(&host.device).expect("DxrBackend::new");
        let rq = DxrRayQueryPipeline::new(&host.device, backend.tier()).expect("RayQuery pipeline");

        // Mesh A: 2 indexed triangles forming a unit quad on z = 2.0
        let verts_a = [
            [-1.0f32, -1.0, 2.0],
            [1.0, -1.0, 2.0],
            [1.0, 1.0, 2.0],
            [-1.0, 1.0, 2.0],
        ];
        let indices_a = [0u32, 1, 2, 0, 2, 3];
        let (vb_a, ib_a) = backend
            .create_triangle_buffers(&verts_a, Some(&indices_a))
            .expect("mesh A buffers");
        let ib_a = ib_a.expect("mesh A ib");
        let blas_a = backend
            .build_blas(&DxrTriangleMeshDesc {
                vertex_buffer_gpu: unsafe { vb_a.GetGPUVirtualAddress() },
                vertex_count: verts_a.len() as u32,
                vertex_stride_bytes: 12,
                index_buffer_gpu: Some((
                    unsafe { ib_a.GetGPUVirtualAddress() },
                    indices_a.len() as u32,
                    INDEX_FORMAT_U32,
                )),
                opaque: true,
                allow_update: false,
                allow_compaction: false,
                prefer_fast_trace: true,
            })
            .expect("build BLAS A");

        // Mesh B: non-indexed triangle placed closer at z = 1.0 when transformed + masked.
        let verts_b = [[-0.5f32, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
        let (vb_b, _) = backend
            .create_triangle_buffers(&verts_b, None)
            .expect("mesh B buffers");
        let blas_b = backend
            .build_blas(&DxrTriangleMeshDesc {
                vertex_buffer_gpu: unsafe { vb_b.GetGPUVirtualAddress() },
                vertex_count: verts_b.len() as u32,
                vertex_stride_bytes: 12,
                index_buffer_gpu: None,
                opaque: true,
                allow_update: false,
                allow_compaction: false,
                prefer_fast_trace: true,
            })
            .expect("build BLAS B");

        let tf_a = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let tf_b = [1.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let instances = [
            DxrInstance {
                blas_gpu_va: blas_a.gpu_va,
                transform_3x4: tf_a,
                instance_id: 10,
                mask: 0x01,
                hit_group_offset: 0,
                flags: 0,
            },
            DxrInstance {
                blas_gpu_va: blas_b.gpu_va,
                transform_3x4: tf_b,
                instance_id: 20,
                mask: 0x02,
                hit_group_offset: 0,
                flags: 0,
            },
        ];
        let tlas = backend
            .build_tlas(&instances, false)
            .expect("build TLAS multi-instance");

        let tris_a = [
            [verts_a[0], verts_a[1], verts_a[2]],
            [verts_a[0], verts_a[2], verts_a[3]],
        ];
        let tris_b = [[verts_b[0], verts_b[1], verts_b[2]]];
        let cpu_scene = [
            CpuInstance {
                triangles: &tris_a,
                transform_3x4: tf_a,
                instance_id: 10,
                mask: 0x01,
            },
            CpuInstance {
                triangles: &tris_b,
                transform_3x4: tf_b,
                instance_id: 20,
                mask: 0x02,
            },
        ];

        let rays = [
            // Hits closer instance 20 (mask 0xFF matches both, z=1 beats z=2).
            RayInput::new([0.1, 0.0, 0.0], [0.0, 0.0, 1.0], 0.01, 100.0),
            // Mask 0x01 excludes instance 20 and hits instance 10 triangle 0 or 1.
            RayInput::new([0.1, -0.2, 0.0], [0.0, 0.0, 1.0], 0.01, 100.0).with_mask(0x01),
            // Hits upper triangle of quad A (primitive 1).
            RayInput::new([-0.5, 0.5, 0.0], [0.0, 0.0, 1.0], 0.01, 100.0),
            // Misses off to the side.
            RayInput::new([5.0, 5.0, 0.0], [0.0, 0.0, 1.0], 0.01, 50.0),
            // Misses due to short tmax.
            RayInput::new([0.25, 0.25, 0.0], [0.0, 0.0, 1.0], 0.01, 0.5),
            // Angled ray hitting mesh B at an off-center barycentric point.
            RayInput::new([-0.2, -0.3, 0.0], [0.15, 0.1, 1.0], 0.01, 100.0),
        ];

        let gpu_hits = rq
            .trace_rays(&mut backend, &tlas, &rays)
            .expect("trace_rays GPU");
        assert_eq!(gpu_hits.len(), rays.len());

        for (i, (ray, gpu)) in rays.iter().zip(gpu_hits.iter()).enumerate() {
            let cpu = trace_cpu_reference(&cpu_scene, ray);
            println!("RAY_{i} CPU={cpu:?} GPU={gpu:?}");
            assert_eq!(gpu.hit, cpu.hit, "ray {i} hit mismatch");
            assert!((gpu.t - cpu.t).abs() < 1e-4, "ray {i} t mismatch");
            if cpu.hit == 1 {
                assert!(
                    (gpu.bary_u - cpu.bary_u).abs() < 1e-4
                        && (gpu.bary_v - cpu.bary_v).abs() < 1e-4,
                    "ray {i} bary mismatch"
                );
                assert_eq!(gpu.primitive_index, cpu.primitive_index, "ray {i} prim_id");
                assert_eq!(gpu.instance_id, cpu.instance_id, "ray {i} inst_id");
                assert_eq!(gpu.front_face, cpu.front_face, "ray {i} front_face");
            }
        }
        println!(
            "DXR_RQ_STATS dxil={}B backend={:?}",
            rq.dxil_byte_len(),
            backend.stats()
        );
    }

    #[test]
    fn dxr_blas_refit_and_compaction_preserve_rayquery_parity() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let tier = crate::dxr_proto::probe_tier(&host.device).unwrap_or(0);
        if tier < 11 {
            return;
        }
        let mut backend = DxrBackend::new(&host.device).expect("DxrBackend::new");
        let rq = DxrRayQueryPipeline::new(&host.device, backend.tier()).expect("RayQuery pipeline");

        // 1. Test BLAS refit (`allow_update = true`, `update_blas`).
        let mut verts = [[-1.0f32, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0]];
        let (vb_upd, _) = backend
            .create_triangle_buffers(&verts, None)
            .expect("vb_upd");
        let mesh_desc = DxrTriangleMeshDesc {
            vertex_buffer_gpu: unsafe { vb_upd.GetGPUVirtualAddress() },
            vertex_count: 3,
            vertex_stride_bytes: 12,
            index_buffer_gpu: None,
            opaque: true,
            allow_update: true,
            allow_compaction: false,
            prefer_fast_trace: false,
        };
        let mut blas_upd = backend
            .build_blas(&mesh_desc)
            .expect("build updatable BLAS");
        let mut tlas = backend
            .build_tlas(&[DxrInstance::identity(blas_upd.gpu_va, 7)], true)
            .expect("build updatable TLAS");

        let probe_ray = [RayInput::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.01, 100.0)];
        let hit_before = rq.trace_rays(&mut backend, &tlas, &probe_ray).unwrap()[0];
        assert_eq!(hit_before.hit, 1);
        assert!((hit_before.t - 2.0).abs() < 1e-4);

        // Shift triangle from z = 2.0 to z = 4.5, upload vertices, refit BLAS + TLAS.
        for v in &mut verts {
            v[2] = 4.5;
        }
        crate::dxr_proto::upload_slice(&vb_upd, &verts).expect("update vb");
        backend
            .update_blas(&mut blas_upd, &mesh_desc)
            .expect("update_blas");
        backend
            .update_tlas(&mut tlas, &[DxrInstance::identity(blas_upd.gpu_va, 7)])
            .expect("update_tlas");

        let hit_after = rq.trace_rays(&mut backend, &tlas, &probe_ray).unwrap()[0];
        assert_eq!(hit_after.hit, 1);
        assert!(
            (hit_after.t - 4.5).abs() < 1e-4,
            "refit t={} expected 4.5",
            hit_after.t
        );

        // 2. Test BLAS compaction (`allow_compaction = true`, larger mesh so compaction shrinks).
        let mut grid_verts = Vec::new();
        for row in 0..8 {
            for col in 0..8 {
                let x = (col as f32 - 4.0) * 0.5;
                let y = (row as f32 - 4.0) * 0.5;
                grid_verts.push([x, y, 3.0]);
                grid_verts.push([x + 0.4, y, 3.0]);
                grid_verts.push([x + 0.2, y + 0.4, 3.0]);
            }
        }
        let (vb_cmp, _) = backend
            .create_triangle_buffers(&grid_verts, None)
            .expect("vb_cmp");
        let mut blas_cmp = backend
            .build_blas(&DxrTriangleMeshDesc {
                vertex_buffer_gpu: unsafe { vb_cmp.GetGPUVirtualAddress() },
                vertex_count: grid_verts.len() as u32,
                vertex_stride_bytes: 12,
                index_buffer_gpu: None,
                opaque: true,
                allow_update: false,
                allow_compaction: true,
                prefer_fast_trace: true,
            })
            .expect("build compactable BLAS");
        let uncompacted = blas_cmp.bytes;

        let tlas_pre = backend
            .build_tlas(&[DxrInstance::identity(blas_cmp.gpu_va, 99)], false)
            .expect("build TLAS pre-compact");
        let cmp_rays = [
            RayInput::new([0.2, 0.2, 0.0], [0.0, 0.0, 1.0], 0.01, 50.0),
            RayInput::new([-1.8, -1.8, 0.0], [0.0, 0.0, 1.0], 0.01, 50.0),
            RayInput::new([10.0, 10.0, 0.0], [0.0, 0.0, 1.0], 0.01, 50.0),
        ];
        let hits_pre = rq.trace_rays(&mut backend, &tlas_pre, &cmp_rays).unwrap();

        let did_compact = backend.compact_blas(&mut blas_cmp).expect("compact_blas");
        let compacted = blas_cmp.bytes;
        assert!(compacted <= uncompacted);
        let tlas_post = backend
            .build_tlas(&[DxrInstance::identity(blas_cmp.gpu_va, 99)], false)
            .expect("build TLAS post-compact");
        let hits_post = rq.trace_rays(&mut backend, &tlas_post, &cmp_rays).unwrap();
        assert_eq!(hits_pre.len(), hits_post.len());
        for (idx, (a, b)) in hits_pre.iter().zip(hits_post.iter()).enumerate() {
            assert_eq!(a.hit, b.hit, "compact ray {idx} hit");
            assert!((a.t - b.t).abs() < 1e-4, "compact ray {idx} t");
            assert_eq!(
                a.primitive_index, b.primitive_index,
                "compact ray {idx} prim"
            );
        }
        println!(
            "DXR_REFIT_COMPACT_RESULT did_compact={did_compact} uncompacted={uncompacted} compacted={compacted} stats={:?}",
            backend.stats()
        );
    }
}
