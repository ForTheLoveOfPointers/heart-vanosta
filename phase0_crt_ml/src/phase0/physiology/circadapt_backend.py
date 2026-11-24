"""pyCircAdapt backend for Phase 0.

This adapter maps the small ``PatientParams`` / ``Setting`` contract onto a
``circadapt.VanOsta2024`` model and extracts the physiology needed by the rest
of the pipeline.  The CircAdapt model is used as a *beat generator*: patient
parameters are applied, the model is run for a few beats, and the last exported
beat is returned as a ``Beat``.

Installation on Linux (the wheel is distributed by the CircAdapt team, not
PyPI)::

    bash scripts/setup_circadapt.sh

or manually::

    python3 -m venv --system-site-packages .venv
    .venv/bin/pip install --no-deps \
        https://framework.circadapt.org/files/circadapt-2602-py3-none-linux_x86_64.whl

The mapping is deliberately explicit and documented because it is the first
thing that must be calibrated in Phase 1:

* contractility and regional viability scale ``Patch.Sf_act``;
* preload uses CircAdapt's volume control (``PFC.is_volume_control``);
* afterload scales the systemic ``ArtVen.k`` and ``ArtVen.p0``;
* programmed AV delay sets ``Timings.tau_av`` (with a fusion approximation for
  patients without AV block);
* VV delay shifts the per-patch activation delays ``Patch.dt``;
* QRS duration modulates the ventricular activation spread and ``Patch.time_act``.

The observation models never see the labels.  They only receive ``t_ms``,
pressure, volume, valve events and activation times.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..types import Beat, BeatLabels, PatientParams, Setting
from .base import Simulator

_PA_TO_MMHG = 133.322387415


def circadapt_available() -> bool:
    try:
        import circadapt  # noqa: F401
    except Exception:
        return False
    return True


def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def _first_true(mask: np.ndarray) -> int:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else 0


def _longest_run(mask: np.ndarray) -> Optional[tuple[int, int]]:
    """Return the longest contiguous run, merging runs that wrap the cycle."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    runs = [
        r
        for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        if r.size
    ]
    if len(runs) > 1 and mask[0] and mask[-1]:
        runs[0] = np.concatenate((runs[-1], runs[0]))
        runs = runs[:-1]
    run = max(runs, key=len)
    return int(run[0]), int(run[-1])


class CircAdaptSimulator(Simulator):
    def __init__(
        self,
        model: str = "VanOsta2024",
        n_beats: int = 3,
        path_to_circadapt: str | None = None,
        reset_every: int = 200,
        vv_penalty_strength: float = 1.0,
    ) -> None:
        if not circadapt_available():
            raise ImportError(
                "pyCircAdapt is not installed. Run "
                "`bash scripts/setup_circadapt.sh` or install the platform "
                "wheel from https://framework.circadapt.org/ into a virtual "
                "environment."
            )

        available = []
        import circadapt

        for name in ("VanOsta2023", "VanOsta2024"):
            if hasattr(circadapt, name):
                available.append(name)
        if not hasattr(circadapt, model):
            raise ValueError(
                f"Unknown CircAdapt model '{model}'. Available: {available}"
            )

        self.model_name = model
        self.n_beats = max(int(n_beats), 1)
        self.path_to_circadapt = path_to_circadapt
        self.reset_every = max(int(reset_every), 1)
        self.vv_penalty_strength = _clip(vv_penalty_strength, 0.0, 1.0)
        self._last_patient_id: int | None = None
        self._n_simulated = 0
        self._model = None
        self._build_model()

    def _build_model(self) -> None:
        import circadapt

        model_cls = getattr(circadapt, self.model_name)
        # pyCircAdapt loads the packaged reference state and, if needed, caches
        # it locally as P_ref_VanOsta2024.npy in the working directory. The
        # cache is regenerable and is not part of the repository.
        self._model = model_cls(path_to_circadapt=self.path_to_circadapt)
        self._model.load_reference()
        self._capture_reference()

    def _ensure_fresh(self, patient: PatientParams) -> None:
        # CircAdapt keeps a small amount of solver/controller state that is not
        # part of the exported reference.  Rebuilding the model per patient and
        # periodically inside a long sweep prevents slow drift from eventually
        # crashing an extreme beat.
        if (
            self._model is None
            or patient.patient_id != self._last_patient_id
            or self._n_simulated >= self.reset_every
        ):
            self._build_model()
            self._last_patient_id = int(patient.patient_id)
            self._n_simulated = 0

    def _capture_reference(self) -> None:
        m = self._model
        self._idx: Dict[str, int] = {
            name: i for i, name in enumerate(m["Patch"].objects_short)
        }
        self._sf_act0 = np.asarray(m["Patch"]["Sf_act"][:], dtype=float).copy()
        self._time_act0 = np.asarray(m["Patch"]["time_act"][:], dtype=float).copy()
        self._artven_k0 = np.asarray(m["ArtVen"]["k"][:], dtype=float).copy()
        self._artven_p00 = np.asarray(m["ArtVen"]["p0"][:], dtype=float).copy()
        self._circ_volume0 = float(
            np.ravel(np.asarray(m["PFC"]["circulation_volume"][:], dtype=float))[0]
        )

    def _apply_patient(self, p: PatientParams) -> None:
        m = self._model
        m.set("Model.t_cycle", 60.0 / _clip(p.hr_bpm, 30.0, 180.0))

        sf = self._sf_act0.copy()
        con = _clip(p.contractility, 0.30, 2.00)
        lv_v = _clip(p.lv_viability, 0.10, 1.00)
        sep_v = _clip(p.septal_viability, 0.10, 1.00)
        lat_v = _clip(p.lateral_viability, 0.10, 1.00)
        rv_v = _clip(p.rv_viability, 0.10, 1.00)

        sf[self._idx["pLv0"]] *= con * lv_v * lat_v
        sf[self._idx["pSv0"]] *= con * lv_v * sep_v
        sf[self._idx["pRv0"]] *= con * rv_v
        m["Patch"]["Sf_act"] = np.clip(sf, 1e3, 3e5)

        # Preload: fix total circulating volume instead of letting the pressure
        # controller compensate every load change.
        m["PFC"]["is_volume_control"] = True
        target = self._circ_volume0 * _clip(p.preload, 0.50, 1.50)
        m["PFC"]["target_volume"] = np.array([target], dtype=float)

        aft = _clip(p.afterload, 0.50, 1.50)
        m["ArtVen"]["k"] = np.array(
            [self._artven_k0[0] * aft, self._artven_k0[1]], dtype=float
        )
        m["ArtVen"]["p0"] = np.array(
            [self._artven_p00[0] * aft, self._artven_p00[1]], dtype=float
        )

    def _apply_setting(self, p: PatientParams, s: Setting) -> None:
        m = self._model

        if p.av_block:
            effective_av = float(s.av_ms)
        else:
            intrinsic = _clip(p.intrinsic_av_ms, 60.0, 400.0)
            # Fusion approximation: if the programmed AV is longer than the
            # intrinsic AV, intrinsic conduction wins progressively.  The small
            # slope keeps a programmed AV beyond the intrinsic value from being
            # perfectly flat, which would make the ranking label ambiguous.
            if s.av_ms <= intrinsic:
                effective_av = float(s.av_ms)
            else:
                effective_av = intrinsic + 0.10 * (float(s.av_ms) - intrinsic)

        # A programmed AV longer than the cycle cannot be stable; keep a
        # conservative filling reserve, although the default patient ranges do
        # not reach this regime.
        rr_ms = 60000.0 / _clip(p.hr_bpm, 30.0, 180.0)
        effective_av = min(effective_av, 0.40 * rr_ms)
        m["Timings"]["law_tau_av"] = 0
        m["Timings"]["tau_av"] = _clip(effective_av, 40.0, 400.0) / 1000.0

        dys = max(
            float(p.dyssynchrony_ms),
            0.5 * max(float(p.qrs_ms) - 100.0, 0.0),
        )
        dys = _clip(dys, 0.0, 150.0)
        base_lv_ms = (
            float(p.lead_lv_delay_ms) - float(p.lead_rv_delay_ms)
        ) + 0.5 * dys
        base_lv = base_lv_ms / 1000.0

        vv = _clip(float(s.vv_ms), -120.0, 120.0) / 1000.0
        dt_lv = base_lv - 0.5 * vv
        dt_rv = 0.5 * vv
        dt_sv = 0.5 * (dt_lv + dt_rv) - 0.15 * dys / 1000.0 - 0.25 * vv

        dt = np.zeros_like(self._sf_act0)
        dt[self._idx["pLv0"]] = dt_lv
        dt[self._idx["pSv0"]] = dt_sv
        dt[self._idx["pRv0"]] = dt_rv
        m["Patch"]["dt"] = np.clip(dt, -0.12, 0.25)

        # Resynchronization efficiency.  CircAdapt resolves regional mechanics,
        # but in this closed-loop, adapted configuration excessive LV
        # pre-excitation does not by itself reduce stroke work.  A smooth
        # penalty around the intrinsic LV-RV delay stands in for the residual
        # dyssynchrony / septal interaction that a real heart shows.  It is a
        # first-order term to be calibrated in Phase 1, not a validated law.
        vv_ms = _clip(float(s.vv_ms), -120.0, 120.0)
        mismatch = (vv_ms - base_lv_ms) / 30.0
        # strength=0 disables the term, strength=1 recovers the Phase 0 term.
        # This makes the VV sensitivity analysis explicit instead of implicit.
        penalty = 1.0 - self.vv_penalty_strength * 0.25 * (
            1.0 - float(np.exp(-0.5 * mismatch * mismatch))
        )
        sf = np.asarray(m["Patch"]["Sf_act"][:], dtype=float)
        sf[self._idx["pLv0"]] *= penalty
        sf[self._idx["pRv0"]] *= penalty
        m["Patch"]["Sf_act"] = np.clip(sf, 1e3, 3e5)

        ta = self._time_act0.copy()
        vent_ta = _clip(
            0.425 + 0.5 * (float(p.qrs_ms) - 150.0) / 1000.0, 0.30, 0.55
        )
        for name in ("pLv0", "pSv0", "pRv0"):
            ta[self._idx[name]] = vent_ta
        m["Patch"]["time_act"] = ta

    def _flow_event(
        self, q: np.ndarray, t_ms: np.ndarray
    ) -> tuple[float, float]:
        """Return the first/last sample of the dominant flow run.

        For a run that wraps the cycle boundary, ``start`` is numerically
        greater than ``end``.  Callers that need a duration use modular
        arithmetic (see ``filling_ms`` in ``_extract_beat``).
        """
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(q)) if q.size else 0.0
        if peak <= 0.0:
            return float(t_ms[0]), float(t_ms[0])
        threshold = max(0.05 * peak, 1e-12)
        run = _longest_run(q > threshold)
        if run is None:
            peak_idx = int(np.argmax(q))
            return float(t_ms[peak_idx]), float(t_ms[peak_idx])
        return float(t_ms[run[0]]), float(t_ms[run[1]])

    def _activation_times(self, t_ms: np.ndarray) -> Dict[str, float]:
        m = self._model
        contraction = np.asarray(m["Patch"]["C"][:], dtype=float)[: len(t_ms)]
        onsets: Dict[str, float] = {}
        for name in ("pLa0", "pRa0", "pLv0", "pSv0", "pRv0"):
            col = contraction[:, self._idx[name]]
            if np.nanmax(col) > 0.01:
                onsets[name] = float(t_ms[_first_true(col > 0.01)])
            else:
                onsets[name] = float(t_ms[0])

        lv = onsets["pLv0"]
        return {
            "la": onsets["pLa0"],
            "ra": onsets["pRa0"],
            "rv": onsets["pRv0"],
            "lv": lv,
            "sep": onsets["pSv0"],
            "lat": lv,
        }

    def _extract_beat(self, patient: PatientParams) -> Beat:
        m = self._model
        t_s = np.asarray(m["Solver"]["t"][:], dtype=float)
        if t_s.size < 2:
            raise RuntimeError("CircAdapt did not export a complete beat.")

        t_ms = t_s * 1000.0
        rr_ms = 60000.0 / _clip(patient.hr_bpm, 30.0, 180.0)
        span_ms = float(t_ms[-1] - t_ms[0])
        if span_ms < 0.5 * rr_ms or span_ms > 1.5 * rr_ms:
            raise RuntimeError(
                "CircAdapt exported an unexpected time span: "
                f"{span_ms:.1f} ms for RR={rr_ms:.1f} ms. "
                "The beat extractor assumes one cardiac cycle."
            )
        volume = np.asarray(m["Cavity"]["V"][:, "cLv"], dtype=float) * 1e6
        pressure = (
            np.asarray(m["Cavity"]["p"][:, "cLv"], dtype=float) / _PA_TO_MMHG
        )
        aortic = (
            np.asarray(m["Cavity"]["p"][:, "SyArt"], dtype=float) / _PA_TO_MMHG
        )

        q_mv = np.asarray(m["Valve"]["q"][:, "LaLv"], dtype=float)

        n = min(
            len(t_s),
            len(volume),
            len(pressure),
            len(aortic),
            len(q_mv),
        )
        t_s, t_ms = t_s[:n], t_ms[:n]
        volume, pressure, aortic = volume[:n], pressure[:n], aortic[:n]
        q_mv = q_mv[:n]
        volume = np.maximum(
            np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0), 1e-6
        )
        pressure = np.nan_to_num(pressure, nan=0.0, posinf=0.0, neginf=0.0)
        aortic = np.nan_to_num(aortic, nan=0.0, posinf=0.0, neginf=0.0)

        mv_open, mv_close = self._flow_event(q_mv, t_ms)
        cycle_ms = float(t_ms[-1] - t_ms[0]) if t_ms[-1] > t_ms[0] else 1.0
        filling_ms = (mv_close - mv_open) % cycle_ms if cycle_ms > 0 else 0.0

        activation = self._activation_times(t_ms)
        spread = max(activation["lv"], activation["sep"], activation["rv"]) - min(
            activation["lv"], activation["sep"], activation["rv"]
        )
        fusion = 1.0 / (1.0 + max(spread, 0.0) / 50.0)

        edv = float(np.max(volume))
        esv = float(np.min(volume))
        ef = (edv - esv) / max(edv, 1e-9)
        stroke_work = float(abs(np.trapezoid(pressure, volume)))
        dpdt_max = float(np.max(np.gradient(pressure, t_s)))

        sf = np.nan_to_num(
            np.asarray(m["Patch"]["Sf"][:], dtype=float)[:n],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        efib = np.nan_to_num(
            np.asarray(m["Patch"]["Ef"][:], dtype=float)[:n],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        work: Dict[str, float] = {}
        for name in ("pLv0", "pSv0", "pRv0"):
            j = self._idx[name]
            work[name] = float(-np.trapezoid(sf[:, j], efib[:, j]))
        w_lv, w_sv = work["pLv0"], work["pSv0"]
        work_asymmetry = (w_lv - w_sv) / (abs(w_lv) + abs(w_sv) + 1e-9)
        wasted_work = _clip(max(0.0, -w_sv) / (abs(w_lv) + 1e-9), 0.0, 2.0)

        labels = BeatLabels(
            stroke_work=stroke_work,
            edv=edv,
            esv=esv,
            ef=ef,
            dpdt_max=dpdt_max,
            work_asymmetry=work_asymmetry,
            wasted_work=wasted_work,
            filling_time_ms=filling_ms,
            fusion_index=fusion,
        )

        return Beat(
            t_ms=t_ms,
            lv_pressure=pressure,
            lv_volume=volume,
            aortic_pressure=aortic,
            activation_times=activation,
            labels=labels,
        )

    def simulate(self, patient: PatientParams, setting: Setting) -> Beat:
        self._ensure_fresh(patient)
        self._model.load_reference()
        self._apply_patient(patient)
        self._apply_setting(patient, setting)
        try:
            self._model.run(self.n_beats)
        except Exception as first_exc:
            # Retry once from a clean model with the same number of beats.  A
            # shorter retry would produce a systematically different label.
            # If the retry also fails, surface both the patient/setting and the
            # original error so the parameter ranges can be tightened.
            self._build_model()
            self._last_patient_id = int(patient.patient_id)
            self._n_simulated = 0
            self._model.load_reference()
            self._apply_patient(patient)
            self._apply_setting(patient, setting)
            try:
                self._model.run(self.n_beats)
            except Exception as exc:
                raise RuntimeError(
                    "CircAdapt crashed for patient_id="
                    f"{patient.patient_id} setting={setting.label()}; "
                    f"first failure was {type(first_exc).__name__}"
                ) from exc

        self._n_simulated += 1
        return self._extract_beat(patient)
