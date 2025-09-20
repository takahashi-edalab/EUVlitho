import numpy as np
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="Mask Generation Script")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output masks",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for mask generation",
    )

    parser.add_argument(
        "--W",
        "-w",
        type=int,
        default=2048,
        help="Mask width.",
    )
    parser.add_argument(
        "--H",
        "-h",
        type=int,
        default=2048,
        help="Mask height.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Threshold: {args.threshold}")
    # Here you would add the logic for mask generation using the provided arguments

    # マスク生成（maskgen関数は別途実装が必要）
    mask2d = np.zeros((args.H, args.W), dtype=bool)
    # maskgen(mask2d, NDIVX, NDIVY)  # 実際のマスク生成

    # バイナリファイル出力
    compressed_bytes = np.packbits(mask2d)
    compressed_bytes.tofile("mask.bin")

    # CSV出力
    mask_2d = mask2d.reshape(args.H, args.W)
    np.savetxt("maskimage.csv", mask_2d, delimiter=",", fmt="%d")


if __name__ == "__main__":
    main()
