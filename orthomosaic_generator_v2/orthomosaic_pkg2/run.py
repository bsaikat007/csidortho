#!/usr/bin/env python3
"""
run.py — Generate an orthomosaic from DJI drone images.

QUICK START:
    python run.py --images /path/to/images/ --output ortho.tif

For 30 images on a 16 GB machine:
    python run.py --images /path/to/images/ --scale 0.5

For low RAM (< 8 GB):
    python run.py --images /path/to/images/ --scale 0.25
"""

import argparse, sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(
        description='DJI Drone Orthomosaic Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --images images/ --output ortho.tif
  python run.py --images images/ --scale 0.25          # low RAM
  python run.py --images images/ --gsd 0.05            # explicit GSD
  python run.py --images images/ --blending multiband  # best seams
  python run.py --images images/ --bundle-adjust       # grid flights only

RAM guide:
  scale=0.50  →  ~1/4 memory usage  (default, good for 8-16 GB)
  scale=0.25  →  ~1/16 memory usage (safe for 4-8 GB)
  scale=0.15  →  ~1/44 memory usage (safe for 2-4 GB)
""")

    p.add_argument('--images',     required=True, type=Path)
    p.add_argument('--output',     default='orthomosaic.tif', type=Path)
    p.add_argument('--gsd',        type=float, default=None,
                   help='Target GSD in metres (default: auto)')
    p.add_argument('--scale',      type=float, default=0.5,
                   help='Image downscale 0.1-1.0 (default: 0.5)')
    p.add_argument('--ground-alt', type=float, default=None,
                   help='Ground elevation m AMSL (default: auto from XMP)')
    p.add_argument('--blending',   choices=['weighted','nearest','multiband'],
                   default='weighted')
    p.add_argument('--bundle-adjust', action='store_true')
    p.add_argument('--min-baseline', type=float, default=2.0,
                   help='Min baseline (m) for frame thinning (0=off)')
    p.add_argument('--agl-tolerance', type=float, default=15.0,
                   help='Max AGL deviation from median (0=off)')
    p.add_argument('--no-overlap-refine', action='store_true',
                   help='Disable overlap XY shift refinement')
    p.add_argument('--use-dem', action='store_true',
                   help='Build DEM for terrain-aware orthorectification')
    p.add_argument('--dem-mode', choices=['sparse','dense','both'],
                   default='sparse', help='DEM point cloud mode')
    p.add_argument('--dem-gsd', type=float, default=None,
                   help='DEM resolution m/px (default: auto)')
    p.add_argument('--dem-strength', type=float, default=0.25,
                   help='DEM strength 0-1 (0=flat, 1=full terrain)')
    args = p.parse_args()

    if not args.images.exists():
        print(f"ERROR: not found: {args.images}"); sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Image dir : {args.images}")
    print(f"Output    : {args.output}")
    if args.gsd:
        print(f"GSD       : {args.gsd} m/px")
    print(f"Scale     : {args.scale}")

    try:
        from orthomosaic import OrthoPipeline
    except ImportError as e:
        print(f"ERROR: {e}\nRun: pip install -r requirements.txt")
        sys.exit(1)

    pipe = OrthoPipeline(
        image_dir   = str(args.images),
        output      = str(args.output),
        gsd         = args.gsd,
        scale       = args.scale,
        ground_alt  = args.ground_alt,
        run_ba      = args.bundle_adjust,
        blending    = args.blending,
        min_baseline_m      = args.min_baseline,
        agl_tolerance_m    = args.agl_tolerance,
        refine_overlap_shift = not args.no_overlap_refine,
        use_dem            = args.use_dem,
        dem_mode           = args.dem_mode,
        dem_gsd            = args.dem_gsd,
        dem_strength       = args.dem_strength,
    )
    try:
        out = pipe.run()
        print(f"\nOpen {out.with_suffix('.kml')} in Google Earth to view.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
