//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by the VIctory build
//! that translated it (the BLAKE3 of our executable), so a rebuilt
//! translator never serves an old translation. Entries go through
//! `victory-pipeline`'s content-addressed cache, which writes them
//! atomically; an entry that does not read back is translated again.
//!
//! The entry is our own encoding of [`HostShader`]: a field added to it
//! must be added to `encode` and `decode` too (entries of older builds
//! are ignored anyway).

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{HostShader, StaticSampler, UserData};
use victory_shader::resource::RootIndexedView;
use victory_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
pub(crate) static TRANSLATED: AtomicU64 = AtomicU64::new(0);
pub(crate) static TRANSLATE_FAILED: AtomicU64 = AtomicU64::new(0);
pub(crate) static TRANSLATE_US: AtomicU64 = AtomicU64::new(0);

/// Keeps translated shaders in `dir` from now on.
pub fn use_shader_cache(dir: PathBuf) {
    match Cache::open(dir) {
        Ok(cache) => {
            let _ = CACHE.set(cache);
        }
        Err(e) => tracing::warn!(error = %e, "no shader cache"),
    }
}

/// Shaders found in the cache and translated since the start.
pub(crate) fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// Shader cache hits, fresh translations, translation failures, and
/// cumulative translation time in microseconds since the start.
pub(crate) fn telemetry() -> [u64; 4] {
    [
        HITS.load(Ordering::Relaxed),
        TRANSLATED.load(Ordering::Relaxed),
        TRANSLATE_FAILED.load(Ordering::Relaxed),
        TRANSLATE_US.load(Ordering::Relaxed),
    ]
}

/// What identifies this translator: the BLAKE3 of its sources, of this
/// file (the entries' encoding) and of the translation settings it runs
/// with (the loop limit, which a diagnostics build can change at run time,
/// and what the host's device said).
fn build() -> &'static [u8; 32] {
    static BUILD: OnceLock<[u8; 32]> = OnceLock::new();
    BUILD.get_or_init(|| {
        let limit = victory_shader::emit::loop_limit();
        if limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(
                limit,
                "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner"
            );
        }
        // The view heap's null slot is part of the translation (it differs
        // by host GPU): found before it goes into the key, so a translation
        // cached without the clamp is never served.
        crate::descriptor::host::probe_view_limit();
        let mut hash = blake3::Hasher::new();
        for source in victory_shader::sources()
            .iter()
            .copied()
            .chain([include_str!("shader_cache.rs")])
        {
            hash.update(&(source.len() as u64).to_le_bytes());
            hash.update(source.as_bytes());
        }
        hash.update(env!("CARGO_PKG_VERSION").as_bytes());
        hash.update(&[u8::from(cfg!(feature = "diagnostics"))]);
        for n in [
            limit,
            victory_shader::emit::view_heap_limit(),
            victory_shader::emit::host_max_wave(),
            // Which group barriers wait is part of the translation too.
            u32::from(victory_shader::emit::group_sync()),
            // And which clean-up passes ran over the translated code.
            victory_shader::emit::shader_opt(),
        ] {
            hash.update(&n.to_le_bytes());
        }
        // So are the switches that pick the form of a wave64 program:
        // without them here a run with `VICTORY_WAVE_WIDE=0` was served
        // the translations an earlier run had cached with the switch unset.
        for name in [
            "VICTORY_WAVE_WIDE",
            "VICTORY_WAVE_WIDE_ONLY",
            "VICTORY_WAVE_DEEP_SKIPS",
            "VICTORY_WAVE_OWN_BITS",
            "VICTORY_F16_NEAREST",
        ] {
            hash.update(name.as_bytes());
            hash.update(b"=");
            if let Some(value) = std::env::var_os(name) {
                hash.update(value.as_encoded_bytes());
            }
            hash.update(&[0]);
        }
        hash.update(b"wave_wide_v1\0");
        *hash.finalize().as_bytes()
    })
}

/// Container hashes (16-digit prefixes) whose wave64 compute program may take
/// the wave-wide form on a 32-lane host (`fxceless/victory-fork#823`, GPL-3.0-or-later):
/// water transmission volumetrics, particle visibility, and SSR tiling.
/// Set `VICTORY_WAVE_WIDE_ONLY=none` or `0` to disable all four.
const WAVE_WIDE_DEFAULT: [&str; 4] = [
    "5dc3b25986622949",
    "6f225b641c5b0708",
    "369cbe906ea72764",
    "2cc4c197ee29e291",
];

fn parse_wave_wide_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|h| h.trim().to_ascii_lowercase())
        .filter(|h| !h.is_empty() && h != "none" && h != "0")
        .collect()
}

fn wave_wide_listed(hash: &str) -> bool {
    static WAVE_WIDE: OnceLock<Vec<String>> = OnceLock::new();
    let list = WAVE_WIDE.get_or_init(|| match std::env::var("VICTORY_WAVE_WIDE_ONLY") {
        Ok(v) => parse_wave_wide_list(&v),
        Err(_) => WAVE_WIDE_DEFAULT.iter().map(|h| (*h).to_owned()).collect(),
    });
    list.iter().any(|h| hash.starts_with(h.as_str()))
}

#[cfg(windows)]
fn translate_listed(c: &[u8]) -> Result<HostShader, String> {
    let hash = blake3::hash(c).to_hex();
    victory_shader::emit::with_wave_wide(wave_wide_listed(&hash), || {
        victory_shader::emit::translate(c)
    })
    .map_err(|e| e.to_string())
}

/// Origin of a host shader container translation.
#[cfg(windows)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShaderOrigin {
    Native,
    StandIn(crate::render_graph::StandInKind),
}

/// Translates `container` (via disk cache when available) and reports whether
/// it resolved from disk cache (`true`) plus native vs. stand-in provenance.
#[cfg(windows)]
fn host_shader_cached_inner(container: &[u8]) -> Result<(HostShader, bool, ShaderOrigin), String> {
    if let Some((s, kind)) = stand_in_with_origin(container) {
        return Ok((s, false, ShaderOrigin::StandIn(kind)));
    }
    let translate = |c: &[u8]| {
        let started = std::time::Instant::now();
        let res = translate_listed(c);
        TRANSLATE_US.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        match &res {
            Ok(_) => {
                TRANSLATED.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                TRANSLATE_FAILED.fetch_add(1, Ordering::Relaxed);
            }
        }
        res
    };
    let r = match CACHE.get() {
        Some(cache) => cached(cache, build(), container, translate),
        None => translate(container).map(|shader| (shader, false)),
    };
    match r {
        Ok((shader, from_cache)) => Ok((shader, from_cache, ShaderOrigin::Native)),
        Err(e) => exposure_stub(container)
            .map(|s| {
                (
                    s,
                    false,
                    ShaderOrigin::StandIn(crate::render_graph::StandInKind::Exposure),
                )
            })
            .ok_or(e),
    }
}

/// Like [`host_shader`], also reporting whether the translation came from the
/// disk cache (`true`) or was freshly translated (`false`).
#[cfg(windows)]
pub(crate) fn host_shader_cached(container: &[u8]) -> Result<(HostShader, bool), String> {
    host_shader_cached_inner(container).map(|(shader, from_cache, _)| (shader, from_cache))
}

/// Translates `container` (via disk cache when available) and reports native vs.
/// stand-in provenance.
#[cfg(windows)]
pub(crate) fn host_shader_with_origin(
    container: &[u8],
) -> Result<(HostShader, ShaderOrigin), String> {
    host_shader_cached_inner(container).map(|(shader, _, origin)| (shader, origin))
}

/// Optional stand-ins from `victory_shader::emit::stand_in`:
/// - `VICTORY_AMBIENT_STUB=1`: replaces diffuse GI temporal resolve passes with
///   a constant ambient light floor (`VICTORY_AMBIENT_STUB_VALUE`, default 0.1).
/// - `VICTORY_SHADOWMASK_STUB=1`: replaces RT distant shadow passes with 1.0
///   (lit) (`VICTORY_SHADOWMASK_STUB_VALUE`, default 1.0).
#[cfg(windows)]
fn stand_in_with_origin(
    container: &[u8],
) -> Option<(HostShader, crate::render_graph::StandInKind)> {
    use victory_shader::emit::stand_in as s;
    let hash = blake3::hash(container).to_hex()[..16].to_owned();
    let kinds = [
        (
            "VICTORY_AMBIENT_STUB",
            crate::render_graph::StandInKind::Ambient,
            s::AMBIENT_HASHES,
            s::AMBIENT_VALUE,
            s::ambient_slots(&hash),
        ),
        (
            "VICTORY_SHADOWMASK_STUB",
            crate::render_graph::StandInKind::ShadowMask,
            s::SHADOWMASK_HASHES,
            s::SHADOWMASK_VALUE,
            s::SHADOWMASK_SLOTS,
        ),
    ];
    for (name, kind, hashes, value, defaults) in kinds {
        if std::env::var(name).ok().as_deref().map(str::trim) != Some("1") {
            continue;
        }
        let listed = match std::env::var(format!("{name}_HASHES")) {
            Ok(v) => v
                .split(|c: char| c == ',' || c.is_whitespace())
                .any(|h| h.eq_ignore_ascii_case(&hash)),
            Err(_) => hashes.contains(&hash.as_str()),
        };
        if !listed {
            continue;
        }
        let value = std::env::var(format!("{name}_VALUE"))
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .unwrap_or(value);
        let slots = std::env::var(format!("{name}_SLOTS")).unwrap_or_default();
        let plan = s::Plan::new(value)
            .with_overrides(defaults)
            .with_overrides(&slots);
        match s::storing_plan(container, plan) {
            Ok((shader, plan)) => {
                tracing::warn!(
                    %hash,
                    stand_in = name,
                    value,
                    outputs = ?plan.seen,
                    "pass replaced by a stand-in writing each output in its own encoding"
                );
                return Some((shader, kind));
            }
            Err(e) => {
                tracing::warn!(%hash, stand_in = name, error = %e, "stand-in does not translate");
            }
        }
    }
    None
}

/// `VICTORY_EXPOSURE_STUB=1`: replaces untranslatable auto-exposure shader
/// with a constant mid-grey luminance stand-in.
#[cfg(windows)]
fn exposure_stub(container: &[u8]) -> Option<HostShader> {
    use victory_shader::emit::exposure;
    if std::env::var("VICTORY_EXPOSURE_STUB").ok()?.trim() != "1" {
        return None;
    }
    let hash = blake3::hash(container).to_hex()[..16].to_owned();
    let listed = match std::env::var("VICTORY_EXPOSURE_STUB_HASHES") {
        Ok(v) => v
            .split(|c: char| c == ',' || c.is_whitespace())
            .any(|h| h.eq_ignore_ascii_case(&hash)),
        Err(_) => exposure::HASHES.contains(&hash.as_str()),
    };
    if !listed {
        return None;
    }
    let value = std::env::var("VICTORY_EXPOSURE_STUB_VALUE")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .unwrap_or(exposure::VALUE);
    match exposure::stub(container, value) {
        Ok(s) => {
            tracing::warn!(%hash, value, "exposure shader replaced by a constant stand-in");
            Some(s)
        }
        Err(e) => {
            tracing::warn!(%hash, error = %e, "exposure stand-in does not translate either");
            None
        }
    }
}

/// The cache entry's converter: the build and the container in, the
/// encoded translation out. `ran` tells a miss from a hit.
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        let shader = (self.translate)(&input[32..]).map_err(anyhow::Error::msg)?;
        Ok(encode(&shader))
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<(HostShader, bool), String> {
    let input = [&build[..], container].concat();
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
    };
    for _ in 0..2 {
        let path = cache
            .get_or_convert(&converter, &input)
            .map_err(|e| format!("{e:#}"))?;
        if let Some(shader) = std::fs::read(&path).ok().and_then(|b| decode(&b)) {
            let was_cached = !converter.ran.load(Ordering::Relaxed);
            let (counter, what) = if was_cached {
                (&HITS, "from the cache")
            } else {
                (&MISSES, "translated and cached")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return Ok((shader, was_cached));
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    (converter.translate)(container).map(|shader| (shader, false))
}

const MAGIC: &[u8; 4] = b"VSH1";
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists.
fn encode(s: &HostShader) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, s.dxil.len() as u32);
    out.extend(&s.dxil);
    word(&mut out, s.user_data.len() as u32);
    for u in &s.user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, s.samplers.len() as u32);
    for sampler in &s.samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, s.static_samplers.len() as u32);
    for t in &s.static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [&s.root_cbvs, &s.root_srvs, &s.root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(&mut out, s.root_indexed_views.len() as u32);
    for view in &s.root_indexed_views {
        for value in [
            u32::from(view.table_slot),
            u32::from(view.constant_slot),
            view.constant_dword,
            view.byte_offset,
            u32::from(view.shift),
        ] {
            word(&mut out, value);
        }
    }
    out
}

fn decode(bytes: &[u8]) -> Option<HostShader> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let n = r.word()? as usize;
    let mut root_indexed_views = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        root_indexed_views.push(RootIndexedView {
            table_slot: u8::try_from(r.word()?).ok()?,
            constant_slot: u8::try_from(r.word()?).ok()?,
            constant_dword: r.word()?,
            byte_offset: r.word()?,
            shift: u8::try_from(r.word()?).ok().filter(|&s| s < 32)?,
        });
    }
    r.0.is_empty().then_some(HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        root_indexed_views,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: vec![1],
            root_indexed_views: vec![RootIndexedView {
                table_slot: 1,
                constant_slot: 2,
                constant_dword: 3,
                byte_offset: 32,
                shift: 5,
            }],
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(
            cached(&cache, &build, container, translate),
            Ok((shader(), false))
        );
        assert_eq!(
            cached(&cache, &build, container, translate),
            Ok((shader(), true))
        );
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(
            cached(&cache, &[8; 32], container, translate),
            Ok((shader(), false))
        );
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH1 broken").unwrap();
        }
        assert_eq!(
            cached(&cache, &build, container, translate),
            Ok((shader(), false))
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(
            cached(&cache, &build, container, translate),
            Ok((shader(), true))
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Failures are not stored.
        let fail = |_: &[u8]| Err::<HostShader, _>("unsupported".to_owned());
        assert!(cached(&cache, &build, b"other", fail).is_err());
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&shader());
        assert_eq!(decode(&bytes), Some(shader()));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
    }

    #[test]
    fn wave_wide_list_parses_defaults_and_rollback_tokens() {
        assert_eq!(WAVE_WIDE_DEFAULT.len(), 4);
        assert!(parse_wave_wide_list("none").is_empty());
        assert!(parse_wave_wide_list("0").is_empty());
        assert!(parse_wave_wide_list("  , NONE , 0 ").is_empty());
        assert_eq!(
            parse_wave_wide_list("5DC3B25986622949, 369cbe906ea72764 "),
            vec!["5dc3b25986622949".to_owned(), "369cbe906ea72764".to_owned()]
        );
        for hash in WAVE_WIDE_DEFAULT {
            assert!(
                wave_wide_listed(hash),
                "missing default wave-wide hash {hash}"
            );
        }
        assert!(!wave_wide_listed("0000000000000000"));
    }
}
