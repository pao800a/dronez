"""Domain randomization (ADR-0006 §7): seeded, config-driven, always written from the nominal.

``DomainRandomizer.draw(rng, d)`` consumes a **fixed-length** block of uniforms
(``rng.random((N_SLOTS, 2))``) once per reset, independent of ``d``. A fixed slot table maps each
parameter to its row (value uniform, extreme-flag uniform); new parameters are appended, never
reordered. The unit variate of a slot is ``z = 2 U - 1``, replaced by ``sign(z)`` with probability
``p_extreme`` so corners are actually visited. Therefore a seed perturbs in the same direction at
every width ``d`` and ``d = 0`` reproduces the nominal exactly.

Parameters are applied as ``nominal * (1 + d * h * z)`` unless stated otherwise (widths in
``DRConfig``). ``apply`` goes through ``DroneSim.set_params``, which writes from the nominal
snapshot, so randomization never drifts across episodes. Rotor drag and ground effect stay off.

Wind and impulse magnitudes come from ``rng_dist`` (``draw_disturbance``), not from ``rng_dr``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from dronez.sim.disturbances import DisturbanceConfig
from dronez.sim.motor_model import MotorParams
from dronez.sim.plant import DroneSim, PhysParams, inertia_triangle_ok, project_inertia_triangle
from dronez.sim.sensors import ImuNoiseConfig

F64 = NDArray[np.float64]

#: Slot table: parameter -> row of the uniform block. Append only, never reorder (ADR-0006 §7).
SLOTS: tuple[str, ...] = (
    "mass",
    "inertia_common",
    "inertia_a0",
    "inertia_a1",
    "inertia_a2",
    "com_x",
    "com_y",
    "com_z",
    "kf_common",
    "kf_m0",
    "kf_m1",
    "kf_m2",
    "kf_m3",
    "km",
    "tau_common",
    "tau_m0",
    "tau_m1",
    "tau_m2",
    "tau_m3",
    "latency",
    "battery",
    "noise",
)
SLOT: dict[str, int] = {name: i for i, name in enumerate(SLOTS)}
N_SLOTS = len(SLOTS)
N_DR = 2 * N_SLOTS
#: Length of the ``rng_dist`` block drawn at every reset (wind, gusts, impulses).
N_DIST = 8


class DRConfig(BaseModel):
    """Full widths (d = 1). Defaults are the ADR-0006 §7 table."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    p_extreme: float = Field(default=0.15, ge=0.0, le=1.0)
    mass: float = 0.20
    inertia_common: float = 0.20
    inertia_aniso: float = 0.05
    com_mm: tuple[float, float, float] = (3.0, 3.0, 5.0)
    k_f: float = 0.15
    k_f_mismatch: float = 0.05
    k_m: float = 0.20
    tau_log2: float = 1.0  # tau = tau_nom * 2^(d z): 15..60 ms
    tau_mismatch: float = 0.10
    latency_s: float = 0.020
    battery_low: float = 0.15
    battery_high: float = 0.05
    noise_log2: float = 1.0  # sensor noise scale 2^(d z)
    wind_mean_m_s: float = 3.0
    gust_std_m_s: float = 1.0
    impulse_max_count: int = 2
    impulse_force_n: float = 2.0
    impulse_torque_nm: float = 0.12
    impulse_duration_s: float = 0.02


@dataclass(frozen=True)
class DRDraw:
    """One DR draw: the plant parameters plus the per-episode noise scale and unit variates."""

    phys: PhysParams
    noise_scale: float
    z: F64 = field(repr=False)  # (N_SLOTS,) unit variates after the extreme rule
    d: float = 0.0
    latency_s: float = 0.0


def scale_noise(cfg: ImuNoiseConfig, scale: float) -> ImuNoiseConfig:
    """Every noise density, bias std and random-walk density times ``scale``."""
    if scale == 1.0:
        return cfg
    return ImuNoiseConfig(**{k: v * scale for k, v in cfg.model_dump().items()})


class DomainRandomizer:
    """Draws ``PhysParams`` (and noise scale) from a nominal snapshot at DR width ``d``."""

    def __init__(
        self,
        nominal: PhysParams,
        cfg: DRConfig | None = None,
        physics_dt_s: float = 0.001,
    ) -> None:
        self.nominal = nominal
        self.cfg = cfg if cfg is not None else DRConfig()
        self.physics_dt_s = float(physics_dt_s)

    # -- sampling ---------------------------------------------------------------------------
    def unit_variates(self, rng: np.random.Generator) -> tuple[F64, F64]:
        """``(z, u)``: unit variates in [-1, 1] (extreme rule applied) and the value uniforms."""
        block = rng.random((N_SLOTS, 2))
        u = block[:, 0]
        z = 2.0 * u - 1.0
        extreme = block[:, 1] < self.cfg.p_extreme
        z = np.where(extreme, np.where(z >= 0.0, 1.0, -1.0), z)
        u = np.where(extreme, (z + 1.0) / 2.0, u)
        return z, u

    def draw(self, rng: np.random.Generator, d: float) -> DRDraw:
        """Consume the fixed block from ``rng`` and build the parameters at width ``d``."""
        z, u = self.unit_variates(rng)
        c, nom = self.cfg, self.nominal

        def zs(name: str) -> float:
            return float(z[SLOT[name]])

        mass = nom.mass_kg * (1.0 + d * c.mass * zs("mass"))

        f_c = 1.0 + d * c.inertia_common * zs("inertia_common")
        delta = (
            d * c.inertia_aniso * np.array([zs("inertia_a0"), zs("inertia_a1"), zs("inertia_a2")])
        )
        inertia = nom.inertia_diag_kg_m2 * f_c * (1.0 + delta)
        for _ in range(10):
            if inertia_triangle_ok(inertia):
                break
            delta = d * c.inertia_aniso * rng.uniform(-1.0, 1.0, 3)
            inertia = nom.inertia_diag_kg_m2 * f_c * (1.0 + delta)
        else:
            inertia = project_inertia_triangle(inertia)

        com = d * np.array([zs("com_x"), zs("com_y"), zs("com_z")]) * np.array(c.com_mm) * 1e-3

        kf_mis = np.array([zs(f"kf_m{i}") for i in range(4)])
        kf_scale = (1.0 + d * c.k_f * zs("kf_common")) * (1.0 + d * c.k_f_mismatch * kf_mis)
        km_scale = 1.0 + d * c.k_m * zs("km")
        tau_mis = np.array([zs(f"tau_m{i}") for i in range(4)])
        tau_scale = 2.0 ** (d * c.tau_log2 * zs("tau_common")) * (
            1.0 + d * c.tau_mismatch * tau_mis
        )
        nm = nom.motor
        motor = MotorParams(
            k_f=nm.k_f * kf_scale,
            k_m=nm.k_m * kf_scale * km_scale,  # k_m/k_f scales with km only
            tau_m_s=nm.tau_m_s * tau_scale,
            omega_min=nm.omega_min.copy(),
            omega_max=nm.omega_max.copy(),
        )
        latency_s = d * c.latency_s * float(u[SLOT["latency"]])
        latency_steps = round(latency_s / self.physics_dt_s)
        battery = (
            1.0
            - c.battery_low * d
            + (c.battery_low + c.battery_high) * d * float(u[SLOT["battery"]])
        )
        noise_scale = 2.0 ** (d * c.noise_log2 * zs("noise"))
        phys = PhysParams(
            mass_kg=float(mass),
            com_offset_b_m=com,
            inertia_diag_kg_m2=np.asarray(inertia, dtype=np.float64),
            motor=motor,
            latency_steps=latency_steps,
            battery_scale=float(battery),
        )
        return DRDraw(phys, float(noise_scale), z, float(d), latency_steps * self.physics_dt_s)

    def sample(self, rng: np.random.Generator, d: float) -> PhysParams:
        """``draw(rng, d).phys``."""
        return self.draw(rng, d).phys

    @staticmethod
    def apply(sim: DroneSim, params: PhysParams) -> None:
        """Write ``params`` into the plant (from the nominal snapshot). Call ``sim.reset`` after."""
        sim.set_params(params)

    # -- helpers for the eval layer ---------------------------------------------------------
    def make_params(
        self,
        *,
        mass: float = 1.0,
        inertia: float = 1.0,
        com_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
        k_f: float | tuple[float, float, float, float] = 1.0,
        k_m: float = 1.0,
        tau_m_s: float | None = None,
        latency_s: float = 0.0,
        battery: float = 1.0,
    ) -> PhysParams:
        """Fixed (non-random) parameters as scale factors of the nominal (the G6 configs)."""
        nom = self.nominal
        kf_scale = np.broadcast_to(np.asarray(k_f, dtype=np.float64), (4,)).copy()
        nm = nom.motor
        tau = nm.tau_m_s if tau_m_s is None else np.full(4, float(tau_m_s))
        motor = MotorParams(
            k_f=nm.k_f * kf_scale,
            k_m=nm.k_m * kf_scale * k_m,
            tau_m_s=np.asarray(tau, dtype=np.float64).copy(),
            omega_min=nm.omega_min.copy(),
            omega_max=nm.omega_max.copy(),
        )
        return PhysParams(
            mass_kg=nom.mass_kg * mass,
            com_offset_b_m=np.asarray(com_mm, dtype=np.float64) * 1e-3,
            inertia_diag_kg_m2=nom.inertia_diag_kg_m2 * inertia,
            motor=motor,
            latency_steps=round(latency_s / self.physics_dt_s),
            battery_scale=float(battery),
        )


def draw_disturbance(
    rng_dist: np.random.Generator, cfg: DRConfig, d: float, horizon_s: float, enabled: bool
) -> DisturbanceConfig:
    """Wind and impulses for one episode (levels >= 4). Always consumes ``N_DIST`` uniforms."""
    u = rng_dist.random(N_DIST)
    if not enabled or d <= 0.0:
        return DisturbanceConfig()
    mag = cfg.wind_mean_m_s * d * float(u[0])
    phi = 2.0 * np.pi * float(u[1])
    n_imp = min(int(u[3] * (cfg.impulse_max_count + 1)), cfg.impulse_max_count)
    return DisturbanceConfig(
        wind_mean_w_m_s=(float(mag * np.cos(phi)), float(mag * np.sin(phi)), 0.0),
        wind_gust_std_m_s=cfg.gust_std_m_s * d * float(u[2]),
        random_impulse_count=n_imp,
        random_impulse_horizon_s=horizon_s,
        random_impulse_duration_s=cfg.impulse_duration_s,
        random_impulse_force_n=cfg.impulse_force_n * d * float(u[4]),
        random_impulse_torque_nm=cfg.impulse_torque_nm * d * float(u[5]),
    )
