// Authoritative ocean/river/pond wave model.
//
// This file is compiled into the world crate *and* included by both build
// scripts. The formula is represented once as an expression tree; the world
// build script emits Rust from it and the render build script emits the GLSL
// include compiled by every water shader. Changing a coefficient or a term
// here therefore changes the CPU field and every GPU water path together.
//
// Keep this file dependency-free: `crates/world/build.rs` and
// `crates/render/build.rs` both `include!` it.

/// One node in the language-neutral wave expression tree.
#[derive(Clone, Debug)]
pub enum WaterExpr {
    Const(f32),
    X,
    Z,
    T,
    Amplitude,
    Angle,
    Rapids,
    Add(Box<WaterExpr>, Box<WaterExpr>),
    Sub(Box<WaterExpr>, Box<WaterExpr>),
    Mul(Box<WaterExpr>, Box<WaterExpr>),
    Sin(Box<WaterExpr>),
    Cos(Box<WaterExpr>),
    Sqrt(Box<WaterExpr>),
    Var(&'static str),
}

#[allow(dead_code)]
impl WaterExpr {
    pub fn c(v: f32) -> Self {
        Self::Const(v)
    }
    pub fn x() -> Self {
        Self::X
    }
    pub fn z() -> Self {
        Self::Z
    }
    pub fn t() -> Self {
        Self::T
    }
    pub fn amplitude() -> Self {
        Self::Amplitude
    }
    pub fn angle() -> Self {
        Self::Angle
    }
    pub fn rapids() -> Self {
        Self::Rapids
    }
    pub fn var(name: &'static str) -> Self {
        Self::Var(name)
    }
    pub fn add(self, rhs: Self) -> Self {
        Self::Add(Box::new(self), Box::new(rhs))
    }
    pub fn sub(self, rhs: Self) -> Self {
        Self::Sub(Box::new(self), Box::new(rhs))
    }
    pub fn mul(self, rhs: Self) -> Self {
        Self::Mul(Box::new(self), Box::new(rhs))
    }
    pub fn sin(self) -> Self {
        Self::Sin(Box::new(self))
    }
    pub fn cos(self) -> Self {
        Self::Cos(Box::new(self))
    }
    pub fn sqrt(self) -> Self {
        Self::Sqrt(Box::new(self))
    }

    /// Render this expression as Rust or GLSL. `lang` only changes numeric
    /// literal suffixes; the structure is shared by both backends.
    pub fn emit(&self, lang: WaveLang) -> String {
        match self {
            Self::Const(v) => match lang {
                WaveLang::Rust => format!("{v:?}f32"),
                WaveLang::Glsl => format!("{v:?}"),
            },
            Self::X => "x".to_string(),
            Self::Z => "z".to_string(),
            Self::T => "t".to_string(),
            Self::Amplitude => "amplitude".to_string(),
            Self::Angle => "angle".to_string(),
            Self::Rapids => "rapids".to_string(),
            Self::Var(name) => (*name).to_string(),
            Self::Add(a, b) => format!("({} + {})", a.emit(lang), b.emit(lang)),
            Self::Sub(a, b) => format!("({} - {})", a.emit(lang), b.emit(lang)),
            Self::Mul(a, b) => format!("({} * {})", a.emit(lang), b.emit(lang)),
            Self::Sin(a) => match lang {
                WaveLang::Rust => format!("({}).sin()", a.emit(lang)),
                WaveLang::Glsl => format!("sin({})", a.emit(lang)),
            },
            Self::Cos(a) => match lang {
                WaveLang::Rust => format!("({}).cos()", a.emit(lang)),
                WaveLang::Glsl => format!("cos({})", a.emit(lang)),
            },
            Self::Sqrt(a) => match lang {
                WaveLang::Rust => format!("({}).sqrt()", a.emit(lang)),
                WaveLang::Glsl => format!("sqrt({})", a.emit(lang)),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveLang {
    Rust,
    Glsl,
}

/// A formula as a sequence of named locals followed by a return expression.
pub struct WaterFormula {
    pub locals: Vec<(&'static str, WaterExpr)>,
    pub body: WaterExpr,
}

#[derive(Clone, Copy)]
pub struct OceanWave {
    pub wavelength: f32,
    pub weight: f32,
    pub angle_offset: f32,
    pub bend_scale: f32,
    pub phase: f32,
    pub warp_scale: f32,
    pub chop_scale: f32,
}

pub const WATER_TAU: f32 = 6.283_185_5;
pub const WATER_G: f32 = 9.81;

/// Six swells spread over roughly a half-turn of headings. Comparable weights
/// keep any one train from reading as a row of parallel ridges; the shortest
/// pair adds the small-scale chop that was missing from the old four-wave
/// field.
pub const OCEAN_WAVES: [OceanWave; 6] = [
    OceanWave {
        wavelength: 31.0,
        weight: 0.23,
        angle_offset: -0.15,
        bend_scale: 0.85,
        phase: 1.3,
        warp_scale: 0.8,
        chop_scale: 0.2,
    },
    OceanWave {
        wavelength: 19.0,
        weight: 0.20,
        angle_offset: 0.95,
        bend_scale: -0.7,
        phase: 4.1,
        warp_scale: -0.6,
        chop_scale: -0.3,
    },
    OceanWave {
        wavelength: 12.0,
        weight: 0.17,
        angle_offset: -0.85,
        bend_scale: 0.55,
        phase: 2.2,
        warp_scale: 0.45,
        chop_scale: 0.45,
    },
    OceanWave {
        wavelength: 7.6,
        weight: 0.14,
        angle_offset: 0.35,
        bend_scale: -0.4,
        phase: 5.2,
        warp_scale: -0.35,
        chop_scale: -0.5,
    },
    OceanWave {
        wavelength: 4.8,
        weight: 0.11,
        angle_offset: 1.55,
        bend_scale: 0.3,
        phase: 0.8,
        warp_scale: 0.3,
        chop_scale: 0.55,
    },
    OceanWave {
        wavelength: 3.2,
        weight: 0.08,
        angle_offset: -1.65,
        bend_scale: -0.2,
        phase: 3.4,
        warp_scale: -0.2,
        chop_scale: -0.4,
    },
];

/// Shared ocean spectrum: six differently-directed swells and a slow
/// bend/warp/envelope. The locals are emitted once so the generated shader
/// evaluates the trigonometric terms once per fragment/vertex.
pub fn ocean_formula() -> WaterFormula {
    let bend = WaterExpr::c(0.32)
        .mul(
            WaterExpr::x()
                .mul(WaterExpr::c(0.011))
                .add(WaterExpr::z().mul(WaterExpr::c(0.007)))
                .add(WaterExpr::c(0.4))
                .sin(),
        )
        .add(
            WaterExpr::c(0.18).mul(
                WaterExpr::x()
                    .mul(WaterExpr::c(0.027))
                    .sub(WaterExpr::z().mul(WaterExpr::c(0.019)))
                    .add(WaterExpr::c(1.9))
                    .sin(),
            ),
        );
    let warp = WaterExpr::c(0.55)
        .mul(
            WaterExpr::x()
                .mul(WaterExpr::c(0.017))
                .add(WaterExpr::z().mul(WaterExpr::c(0.013)))
                .add(WaterExpr::c(0.8))
                .sin(),
        )
        .add(
            WaterExpr::c(0.25).mul(
                WaterExpr::x()
                    .mul(WaterExpr::c(0.031))
                    .sub(WaterExpr::z().mul(WaterExpr::c(0.021)))
                    .add(WaterExpr::c(2.0))
                    .sin(),
            ),
        );
    // Higher-frequency phase modulation bends local crests in a way a slow
    // envelope cannot, so wave groups do not line up across the whole view.
    let chop = WaterExpr::c(0.18)
        .mul(
            WaterExpr::x()
                .mul(WaterExpr::c(0.043))
                .add(WaterExpr::z().mul(WaterExpr::c(0.031)))
                .add(WaterExpr::c(1.7))
                .sin(),
        )
        .add(
            WaterExpr::c(0.12).mul(
                WaterExpr::x()
                    .mul(WaterExpr::c(0.067))
                    .sub(WaterExpr::z().mul(WaterExpr::c(0.051)))
                    .add(WaterExpr::c(3.4))
                    .sin(),
            ),
        );
    // Domain warp bends the crest lines over a few blocks. A pure plane-wave
    // sum always re-forms visible rows somewhere in a long view; this makes
    // adjacent crests meander independently instead of staying parallel.
    let warp_x = WaterExpr::c(1.8)
        .mul(
            WaterExpr::z()
                .mul(WaterExpr::c(0.033))
                .add(WaterExpr::x().mul(WaterExpr::c(0.007)))
                .add(WaterExpr::c(2.1))
                .sin(),
        )
        .add(
            WaterExpr::c(1.1).mul(
                WaterExpr::x()
                    .mul(WaterExpr::c(0.051))
                    .sub(WaterExpr::z().mul(WaterExpr::c(0.019)))
                    .add(WaterExpr::c(0.3))
                    .sin(),
            ),
        );
    let warp_z = WaterExpr::c(1.5)
        .mul(
            WaterExpr::x()
                .mul(WaterExpr::c(0.027))
                .add(WaterExpr::z().mul(WaterExpr::c(0.011)))
                .add(WaterExpr::c(1.2))
                .cos(),
        )
        .add(
            WaterExpr::c(0.9).mul(
                WaterExpr::z()
                    .mul(WaterExpr::c(0.043))
                    .sub(WaterExpr::x().mul(WaterExpr::c(0.023)))
                    .add(WaterExpr::c(4.4))
                    .cos(),
            ),
        );
    let envelope = WaterExpr::c(0.80).add(
        WaterExpr::c(0.20)
            .mul(
                WaterExpr::x()
                    .mul(WaterExpr::c(0.009))
                    .add(WaterExpr::z().mul(WaterExpr::c(0.014)))
                    .add(WaterExpr::c(0.7))
                    .sin(),
            )
            .mul(
                WaterExpr::x()
                    .mul(WaterExpr::c(0.021))
                    .sub(WaterExpr::z().mul(WaterExpr::c(0.011)))
                    .add(WaterExpr::c(2.3))
                    .sin(),
            ),
    );

    let mut locals = vec![
        ("bend", bend),
        ("warp", warp),
        ("chop", chop),
        ("warp_x", warp_x),
        ("warp_z", warp_z),
        ("envelope", envelope),
    ];
    let mut height = WaterExpr::c(0.0);
    for (i, wave) in OCEAN_WAVES.iter().enumerate() {
        let a_name: &'static str = ["a1", "a2", "a3", "a4", "a5", "a6"][i];
        let along_name: &'static str =
            ["along1", "along2", "along3", "along4", "along5", "along6"][i];
        locals.push((
            a_name,
            WaterExpr::angle()
                .add(WaterExpr::c(wave.angle_offset))
                .add(WaterExpr::var("bend").mul(WaterExpr::c(wave.bend_scale))),
        ));
        locals.push((
            along_name,
            WaterExpr::x()
                .add(WaterExpr::var("warp_x"))
                .mul(WaterExpr::var(a_name).cos())
                .add(
                    WaterExpr::z()
                        .add(WaterExpr::var("warp_z"))
                        .mul(WaterExpr::var(a_name).sin()),
                ),
        ));
        let k = WATER_TAU / wave.wavelength;
        let omega = (WATER_G * k).sqrt();
        let phase = WaterExpr::c(k)
            .mul(WaterExpr::var(along_name))
            .sub(WaterExpr::c(omega).mul(WaterExpr::t()))
            .add(WaterExpr::c(wave.phase))
            .add(WaterExpr::var("warp").mul(WaterExpr::c(wave.warp_scale)))
            .add(WaterExpr::var("chop").mul(WaterExpr::c(wave.chop_scale)));
        height = height.add(WaterExpr::c(wave.weight).mul(phase.sin()));
    }
    locals.push(("height", height));
    WaterFormula {
        locals,
        body: WaterExpr::amplitude()
            .mul(WaterExpr::var("envelope"))
            .mul(WaterExpr::var("height")),
    }
}

/// Short, fast ripple train aligned with the body's flow; rapids add chop.
pub fn river_formula() -> WaterFormula {
    let along = WaterExpr::x()
        .mul(WaterExpr::angle().cos())
        .add(WaterExpr::z().mul(WaterExpr::angle().sin()));
    let across = WaterExpr::x()
        .mul(WaterExpr::angle().sin())
        .mul(WaterExpr::c(-1.0))
        .add(WaterExpr::z().mul(WaterExpr::angle().cos()));
    let phase = WaterExpr::c(2.6)
        .mul(WaterExpr::var("along"))
        .sub(WaterExpr::c(2.4).add(WaterExpr::amplitude().mul(WaterExpr::c(9.0))).mul(WaterExpr::t()));
    let cross = WaterExpr::c(1.7)
        .mul(WaterExpr::var("across"))
        .sub(WaterExpr::c(1.1).mul(WaterExpr::t()))
        .add(WaterExpr::c(0.8));
    let height = WaterExpr::c(0.62)
        .mul(WaterExpr::var("phase").sin())
        .add(WaterExpr::c(0.38).mul(WaterExpr::var("cross").sin()))
        .add(
            WaterExpr::rapids().mul(WaterExpr::c(0.9)).mul(
                WaterExpr::c(5.7)
                    .mul(WaterExpr::var("along"))
                    .sub(WaterExpr::c(9.0).mul(WaterExpr::t()))
                    .add(WaterExpr::c(2.3).mul(WaterExpr::var("across")))
                    .sin(),
            ),
        )
        .add(
            WaterExpr::rapids().mul(WaterExpr::c(0.4)).mul(
                WaterExpr::c(11.0)
                    .mul(WaterExpr::var("across"))
                    .add(WaterExpr::c(7.0).mul(WaterExpr::t()))
                    .sin(),
            ),
        );
    WaterFormula {
        locals: vec![
            ("along", along),
            ("across", across),
            ("phase", phase),
            ("cross", cross),
            ("height", height),
        ],
        body: WaterExpr::amplitude().mul(WaterExpr::var("height")),
    }
}

/// Faint single ripple for ponds and player-made water.
pub fn pond_formula() -> WaterFormula {
    let along = WaterExpr::x()
        .mul(WaterExpr::angle().cos())
        .add(WaterExpr::z().mul(WaterExpr::angle().sin()));
    let k = WATER_TAU / 5.5;
    let omega = (WATER_G * k).sqrt();
    let phase = WaterExpr::c(k)
        .mul(WaterExpr::var("along"))
        .sub(WaterExpr::c(omega).mul(WaterExpr::t()));
    WaterFormula {
        locals: vec![("along", along), ("phase", phase)],
        body: WaterExpr::amplitude()
            .mul(WaterExpr::c(0.5))
            .mul(WaterExpr::var("phase").sin()),
    }
}
