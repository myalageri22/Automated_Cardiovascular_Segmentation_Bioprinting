"""Helper script to run Phase B CLI easily."""
import sys

from phaseb.src.phaseb.cli import build_parser, main


if __name__ == "__main__":
    # Show a friendly usage hint if run without arguments
    if len(sys.argv) == 1:
        parser = build_parser()
        parser.print_help()
        sys.exit("\nExample: python run_phaseb.py --ct /path/to/ct.nii.gz --seg /path/to/seg.nii.gz --case-id demo\n")
    main()
