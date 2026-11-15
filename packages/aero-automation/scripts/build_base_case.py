"""Build the AERO_AUTOMATION Fluent base case (docs/FLUENT_BASE_CASE.md) automatically with PyFluent.

    python scripts/build_base_case.py MESH.msh OUT.cas.h5 [--chord 0.175] [--speed 19.444] [--elements main,flap1,flap2]

Same settings as the manual guide, except the turbulence model: k-omega SST (the guide allows it when
Transition SST is unstable; set [screening] transition = "turbulent" to match). Every step prints OK or the
error and the names the settings object offers, so a renamed setting in another Fluent version is easy to fix.
"""
import argparse
from pathlib import Path

import ansys.fluent.core as pf


def step(name, fn, obj=None):
    try:
        fn()
        print(f"[ok]   {name}", flush=True)
        return True
    except Exception as exc:  # report and keep going: one bad setting should not lose the session
        hint = f"  (children: {getattr(obj, 'child_names', '')})" if obj is not None else ""
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}{hint}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out")
    ap.add_argument("--chord", type=float, default=0.175)
    ap.add_argument("--speed", type=float, default=19.444)
    ap.add_argument("--elements", default="main,flap1,flap2")
    ap.add_argument("--processes", type=int, default=4)
    a = ap.parse_args()
    els = a.elements.split(",")
    mesh, out = Path(a.mesh).resolve(), Path(a.out).resolve()

    s = pf.launch_fluent(dimension=2, precision="double", processor_count=a.processes, mode="solver",
                         ui_mode="no_gui", cwd=str(out.parent))
    st = s.settings
    try:
        step("read mesh", lambda: st.file.read_mesh(file_name=str(mesh)))
        vis = st.setup.models.viscous
        step("k-omega SST", lambda: (setattr(vis, "model", "k-omega"), setattr(vis, "k_omega_model", "sst")), vis)

        ne = st.setup.named_expressions
        step("named expression speed", lambda: (ne.create(name="speed"), setattr(ne["speed"], "definition", f"{a.speed} [m/s]")), ne)

        vi = st.setup.boundary_conditions.velocity_inlet["inlet"]
        step("inlet speed", lambda: setattr(vi.momentum.velocity_magnitude, "value", "speed"), vi.momentum)
        step("inlet turbulence", lambda: (
            setattr(vi.turbulence, "turbulence_specification", "Intensity and Viscosity Ratio"),
            setattr(vi.turbulence, "turbulent_intensity", 0.005),
            setattr(vi.turbulence, "turbulent_viscosity_ratio", 10)), vi.turbulence)
        po = st.setup.boundary_conditions.pressure_outlet["outlet"]
        step("outlet backflow turbulence", lambda: (
            setattr(po.turbulence, "turbulence_specification", "Intensity and Viscosity Ratio"),
            setattr(po.turbulence, "backflow_turbulent_intensity", 0.005),
            setattr(po.turbulence, "backflow_turbulent_viscosity_ratio", 10)), po.turbulence)

        gw = st.setup.boundary_conditions.wall["ground"].momentum
        step("moving ground", lambda: (
            setattr(gw, "wall_motion", "Moving Wall"),
            setattr(gw, "relative", False),
            setattr(gw.speed, "value", "speed"),
            setattr(gw, "direction", [1, 0])), gw)

        rv = st.setup.reference_values
        step("reference values", lambda: (setattr(rv, "area", a.chord),
                                          setattr(rv, "length", a.chord), setattr(rv, "density", 1.225),
                                          setattr(rv, "velocity", a.speed)), rv)

        m = st.solution.methods
        step("coupled", lambda: setattr(m.p_v_coupling, "flow_scheme", "Coupled"), m.p_v_coupling)
        step("pseudo time", lambda: setattr(m.pseudo_time_method, "formulation", {"coupled_solver": "global-time-step"}),
             m.pseudo_time_method)
        step("second order", lambda: setattr(m.spatial_discretization, "discretization_scheme",
                                             {"pressure": "second-order", "mom": "second-order-upwind",
                                              "k": "second-order-upwind", "omega": "second-order-upwind"}),
             m.spatial_discretization)

        res = st.solution.monitor.residual
        step("residuals 1e-6", lambda: [setattr(res.equations[e], "absolute_criteria", 1e-6)
                                        for e in res.equations.keys()], res)

        fr = st.solution.report_definitions.force
        defs = [("cx", [1, 0], els), ("cy", [0, 1], els)] + [(f"c{d}-{e}", v, [e]) for e in els
                                                              for d, v in (("x", [1, 0]), ("y", [0, 1]))]
        for name, vec, zones in defs:
            step(f"report {name}", lambda name=name, vec=vec, zones=zones: (
                fr.create(name=name),
                setattr(fr[name], "report_output_type", "Force Coefficient"),
                setattr(fr[name], "force_vector", vec),
                setattr(fr[name], "zones", zones)), fr)
        rf = st.solution.monitor.report_files
        step("report file", lambda: (rf.create(name="coef-rfile"),
                                     setattr(rf["coef-rfile"], "report_defs", [d[0] for d in defs]),
                                     setattr(rf["coef-rfile"], "file_name", "coef-rfile.out")), rf)

        con = st.results.graphics.contour
        for name, field in (("contour-pressure", "pressure-coefficient"), ("contour-velocity", "velocity-magnitude")):
            step(name, lambda name=name, field=field: (con.create(name=name), setattr(con[name], "field", field),
                                                       setattr(con[name], "filled", True)), con)
        # saved views: camera on the wing (near) and on the slots (slot); the journal restores them
        tv = s.tui.views
        for name, (tx, ty, w) in (("near", (0.5 * a.chord * 3, 0.2, 4 * a.chord)), ("slot", (a.chord, 0.1, 1.2 * a.chord))):
            step(f"view {name}", lambda name=name, tx=tx, ty=ty, w=w: (
                s.tui.display.mesh(), s.tui.views.camera.target(tx, ty, 0), s.tui.views.camera.position(tx, ty, 10),
                s.tui.views.camera.field(w, w * 0.6), tv.save_view(name)))

        step("init + 20 iterations", lambda: (st.solution.initialization.hybrid_initialize(),
                                              st.solution.run_calculation.iterate(iter_count=20)))
        step("write case", lambda: st.file.write_case(file_name=str(out)))
    finally:
        s.exit()


if __name__ == "__main__":
    main()
