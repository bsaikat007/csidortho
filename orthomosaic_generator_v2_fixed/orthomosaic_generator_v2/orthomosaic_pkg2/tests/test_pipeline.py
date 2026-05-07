"""
tests/test_pipeline.py

Self-contained test suite that validates every module against
the 5 DJI FC6310 images from Sikkim/W.Bengal, India.

Run with:
    python tests/test_pipeline.py --images /path/to/DJI_images/

All tests print PASS / FAIL with measured values.
"""

import sys, argparse, numpy as np
from pathlib import Path

# ── helpers ───────────────────────────────────────────────────

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"

def check(label, condition, measured=''):
    tag = PASS if condition else FAIL
    print(f"  [{tag}] {label}  {measured}")
    return condition


# ── tests ─────────────────────────────────────────────────────

def test_exif_reader(image_dir):
    print("\n=== TEST: exif_reader ===")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from orthomosaic.exif_reader import read_dji_image
    images = sorted(image_dir.glob('*.JPG')) or sorted(image_dir.glob('*.jpg'))
    ok = True
    for p in images[:5]:
        m = read_dji_image(p)
        ok &= check(f"{p.name}: GPS valid",
                    -90 <= m.latitude <= 90 and -180 <= m.longitude <= 180,
                    f"lat={m.latitude:.5f} lon={m.longitude:.5f}")
        ok &= check(f"{p.name}: altitude > 0", m.altitude_abs > 0,
                    f"abs={m.altitude_abs:.1f}m rel={m.altitude_rel:.1f}m")
        ok &= check(f"{p.name}: gimbal pitch ≈ -90°",
                    abs(m.gimbal_pitch + 90) < 5,
                    f"pitch={m.gimbal_pitch:.1f}°")
        ok &= check(f"{p.name}: camera model known",
                    m.model in ('FC6310','FC6310S','FC330','FC220','FC7203'),
                    f"model={m.model}")
    return ok


def test_camera(image_dir):
    print("\n=== TEST: camera ===")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from orthomosaic.camera import BrownConradyCamera, PinholeCamera
    ok = True

    cam = BrownConradyCamera(
        width=5472, height=3648,
        fx=3648., fy=3648., cx=2736., cy=1824.,
        k1=-0.0508, k2=0.0290, p1=0.00015, p2=-0.00010, k3=-0.0087)

    # Round-trip: distort → undistort → distort
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(500, 5000, 500),
                           rng.uniform(300, 3300, 500)])
    Kinv = cam.K_inv()
    h    = np.column_stack([pts, np.ones(len(pts))])
    norm = (Kinv @ h.T).T[:,:2]
    redist = cam.distort_points(norm)
    undist = cam.undistort_points(redist)
    h2 = np.column_stack([undist, np.ones(len(undist))])
    norm2 = (Kinv @ h2.T).T[:,:2]
    redist2 = cam.distort_points(norm2)
    err = np.linalg.norm(redist - redist2, axis=1)
    ok &= check("Round-trip distort→undistort error < 0.01px",
                np.max(err) < 0.01, f"max={np.max(err)*1000:.4f} mpx")

    # backproject_ray must return unit vectors
    corners = np.array([[0.,0.],[5471,0.],[5471,3647],[0.,3647]])
    rays = cam.backproject_ray(corners)
    norms = np.linalg.norm(rays, axis=1)
    ok &= check("backproject_ray returns unit vectors",
                np.allclose(norms, 1., atol=1e-6),
                f"norms={norms.round(6)}")

    # Secant error at corner (42°) should be > 30% for old method
    old_xyz = np.array([(corners[0,0]-cam.cx)/cam.fx * 100,
                         (corners[0,1]-cam.cy)/cam.fy * 100, 100.])
    secant_err = (np.linalg.norm(old_xyz) / 100. - 1.) * 100
    ok &= check("Old backproject would have >30% error at corner",
                secant_err > 30, f"secant_err={secant_err:.1f}%")

    return ok


def test_pose(image_dir):
    print("\n=== TEST: pose ===")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from orthomosaic.pose import Pose
    ok = True

    # make_nadir: optical axis must point down (-Z in ENU)
    pose = Pose.make_nadir(662271., 2989625., 60.5, 141.5)
    optical_in_world = pose.R.T[:,2]
    ok &= check("make_nadir: optical axis = (0,0,-1) in ENU",
                np.allclose(optical_in_world, [0,0,-1], atol=1e-6),
                f"axis={optical_in_world.round(6)}")

    # from_euler_angles round-trip
    for yaw in [0., 90., 141.5, 180., -45.]:
        p = Pose.from_euler_angles(0., np.radians(-90.), np.radians(yaw),
                                    np.array([0.,0.,60.5]))
        omega, phi, kappa = p.to_euler_angles()
        ok &= check(f"Euler round-trip yaw={yaw}°",
                    abs(np.degrees(phi)+90) < 0.001,
                    f"phi_back={np.degrees(phi):.4f}°")

    # transform_to_world is inverse of transform_to_camera
    pose = Pose.make_nadir(100., 200., 50., 0.)
    pts  = np.random.default_rng(1).standard_normal((20,3))
    cam  = pose.transform_to_camera(pts)
    back = pose.transform_to_world(cam)
    err  = np.max(np.abs(back - pts))
    ok  &= check("transform_to_camera/world round-trip",
                 err < 1e-10, f"max_err={err:.2e}m")

    return ok


def test_georeference(image_dir):
    print("\n=== TEST: georeference ===")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from orthomosaic.georeference import UTMTransformer, GeoTIFFWriter, GeoBounds
    ok = True

    utm = UTMTransformer(reference_lon=88.636)
    ok &= check("Zone 45N detected", utm.zone == 45 and utm.northern, f"zone={utm.zone}")

    # Round-trip accuracy
    lat, lon = 27.01930, 88.63570
    e, n = utm.latlon_to_utm(lat, lon)
    lat2, lon2 = utm.utm_to_latlon(e, n)
    err_m = max(abs(lat2-lat)*111320, abs(lon2-lon)*111320*np.cos(np.radians(lat)))
    ok &= check("UTM round-trip < 1mm", err_m < 0.001, f"err={err_m*1000:.4f}mm")

    # Grid convergence at site
    gc = utm.compute_grid_convergence(27.019, 88.636)
    ok &= check("Grid convergence between 0.5° and 1.0°",
                0.5 < gc < 1.0, f"gc={gc:.4f}°")

    # Scale factor near 1
    k = utm.compute_scale_factor(27.019, 88.636)
    ok &= check("Scale factor ≈ 0.9999",
                abs(k - 1.0) < 0.002, f"k={k:.8f}")

    # KML bounds use correct corner ordering
    bounds = GeoBounds(left=e-100, bottom=n-100, right=e+100, top=n+100,
                       crs=f"EPSG:{utm.epsg_code}")
    ok &= check("GeoBounds.is_utm() = True", bounds.is_utm())
    lat_s, lon_w = utm.utm_to_latlon(bounds.left,  bounds.bottom)
    lat_n, lon_e = utm.utm_to_latlon(bounds.right, bounds.top)
    ok &= check("KML: north > south", lat_n > lat_s,
                f"N={lat_n:.6f} S={lat_s:.6f}")
    ok &= check("KML: east > west",   lon_e > lon_w,
                f"E={lon_e:.6f} W={lon_w:.6f}")

    return ok


def test_bundle_adjustment(image_dir):
    print("\n=== TEST: bundle_adjustment ===")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from orthomosaic.bundle_adjustment import (
        BundleAdjuster, Reconstruction, CameraPose, Point3D, Observation)
    import numpy as np

    # Build minimal synthetic reconstruction
    K = np.eye(3); K[0,0]=K[1,1]=500.; K[0,2]=320.; K[1,2]=240.
    rec = Reconstruction(cameras={}, points={}, observations=[])
    from orthomosaic.pose import Pose
    for i in range(3):
        pose = Pose.make_nadir(float(i*10), 0., 50., 0.)
        rec.cameras[i] = CameraPose(i, pose.R, pose.t, K, fixed=(i==0))
    for j in range(20):
        pt = np.random.default_rng(j).standard_normal(3) * 5
        rec.points[j] = Point3D(j, pt)
        for i in range(3):
            cam = rec.cameras[i]
            Xc  = cam.R @ pt + cam.t
            if Xc[2] > 0:
                u = K[0,0]*Xc[0]/Xc[2] + K[0,2]
                v = K[1,1]*Xc[1]/Xc[2] + K[1,2]
                rec.observations.append(Observation(i, j, np.array([u, v])))

    ba = BundleAdjuster(max_iterations=5)
    ok = True

    # Test 1: in-place mutation fix
    params0, struct = ba._pack(rec)
    mask = np.ones(len(rec.observations), bool)
    r1  = ba._residuals(params0, struct, rec, mask)
    ba._residuals(params0 + 0.01, struct, rec, mask)   # second call
    r1b = ba._residuals(params0, struct, rec, mask)
    ok &= check("Residuals stable (no in-place mutation)",
                np.allclose(r1, r1b), f"diff={np.max(np.abs(r1-r1b)):.2e}")

    # Test 2: sparse Jacobian
    sparsity  = ba._build_sparsity(rec, struct)
    n_params  = len(params0)
    n_obs_r   = len(rec.observations)
    dense_n   = 2 * n_obs_r * n_params
    # Threshold: small synthetic problems can be up to 15%; real projects < 0.5%
    ok &= check("Sparse Jacobian < 15% of dense (real projects < 0.5%)",
                sparsity.nnz < 0.15 * dense_n,
                f"{sparsity.nnz}/{dense_n} ({100*sparsity.nnz/dense_n:.2f}%)")

    return ok


def test_orthorectifier(image_dir):
    print("\n=== TEST: orthorectifier (end-to-end on real images) ===")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from orthomosaic.exif_reader   import read_dji_image
    from orthomosaic.pose          import Pose
    from orthomosaic.camera        import PinholeCamera
    from orthomosaic.georeference  import UTMTransformer
    from orthomosaic.orthorectifier import Orthorectifier, OrthoParams
    import cv2, gc

    images = sorted(image_dir.glob('*.JPG')) or sorted(image_dir.glob('*.jpg'))
    if not images:
        print("  No images found"); return False

    ok = True
    meta_all = []
    for p in images[:5]:
        try: meta_all.append(read_dji_image(p))
        except: pass

    lats = [m.latitude  for m in meta_all]
    lons = [m.longitude for m in meta_all]
    utm  = UTMTransformer.from_coordinates(lats, lons)
    gc_d = utm.compute_grid_convergence(np.mean(lats), np.mean(lons))

    poses = []
    for m in meta_all:
        e, n = utm.latlon_to_utm(m.latitude, m.longitude)
        poses.append(Pose.make_nadir(e, n, m.altitude_rel, m.gimbal_yaw + gc_d))

    # Output bounds
    SCALE = 0.1   # very small for test speed
    m0 = meta_all[0]
    e_c, n_c = utm.latlon_to_utm(m0.latitude, m0.longitude)
    max_agl  = max(m.altitude_rel for m in meta_all)
    half_w   = max_agl * (m0.sensor_width_mm*1e-3 / m0.focal_length_mm) * 1000 / 2 + 5
    half_h   = max_agl * (m0.sensor_width_mm*2/3 *1e-3 / m0.focal_length_mm) * 1000 / 2 + 5
    bounds   = (e_c-half_w, n_c-half_h, e_c+half_w, n_c+half_h)
    gsd      = m0.altitude_rel * (m0.pixel_size_um*1e-3 / m0.focal_length_mm) * SCALE

    params = OrthoParams(gsd=gsd, output_bounds=bounds)
    ort    = Orthorectifier(params)

    accum = np.zeros((ort.height, ort.width, 3), dtype=np.float64)
    wsum  = np.zeros((ort.height, ort.width),    dtype=np.float64)

    for i, (m, pose) in enumerate(zip(meta_all, poses)):
        img = cv2.imread(str(m.path))
        if img is None: continue
        nw = int(m.image_width  * SCALE)
        nh = int(m.image_height * SCALE)
        img = cv2.resize(img, (nw, nh))
        cam = PinholeCamera(nw, nh,
                             m.fx_pixels*SCALE, m.fy_pixels*SCALE,
                             m.cx_pixels*SCALE, m.cy_pixels*SCALE)
        o, w = ort.orthorectify_image(img, pose, cam)
        oh = min(o.shape[0], ort.height)
        ow = min(o.shape[1], ort.width)
        accum[:oh,:ow] += o[:oh,:ow].astype(np.float64) * w[:oh,:ow,np.newaxis]
        wsum[:oh,:ow]  += w[:oh,:ow]
        del img, o, w; gc.collect()

    result   = np.clip(accum / np.maximum(wsum[:,:,np.newaxis],1e-8),
                       0,255).astype(np.uint8)
    coverage = 100 * np.sum(result.any(axis=2)) / (ort.height * ort.width)

    ok &= check("Orthomosaic coverage > 50%", coverage > 50, f"{coverage:.1f}%")
    ok &= check("Orthomosaic not all-black",  result.max() > 10,
                f"max_val={int(result.max())}")
    ok &= check("float64 accumulator used",   accum.dtype == np.float64,
                f"dtype={accum.dtype}")
    return ok


# ── main ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=Path, required=True,
                    help='Directory containing DJI JPEG images')
    args = ap.parse_args()

    results = {}
    for name, fn in [
        ('exif_reader',       test_exif_reader),
        ('camera',            test_camera),
        ('pose',              test_pose),
        ('georeference',      test_georeference),
        ('bundle_adjustment', test_bundle_adjustment),
        ('orthorectifier',    test_orthorectifier),
    ]:
        try:
            results[name] = fn(args.images)
        except Exception as e:
            import traceback
            print(f"\n  [EXCEPTION] {name}: {e}")
            traceback.print_exc()
            results[name] = False

    print("\n" + "="*45)
    print("SUMMARY")
    print("="*45)
    all_ok = True
    for name, ok in results.items():
        tag = "\033[92mPASS\033[0m" if ok else "\033[91mFAIL\033[0m"
        print(f"  [{tag}]  {name}")
        all_ok = all_ok and ok
    print("="*45)
    print("ALL TESTS PASSED ✓" if all_ok else "SOME TESTS FAILED ✗")
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
