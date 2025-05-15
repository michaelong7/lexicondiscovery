#!/usr/bin/env python3

import argparse
from glob import glob
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor, HubertForCTC

SEED = 3939

SR = 16000

LAYERS = (
    "z",
    "c0",
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
    "c6",
    "c7",
    "c8",
    "c9",
    "c10",
    "c11",
)


def collate(args: Sequence[dict]) -> dict:
    return {
        "audio": {
            "path": tuple(x["audio"]["path"] for x in args),
            "array": tuple(x["audio"]["array"] for x in args),
            "sampling_rate": tuple(x["audio"]["sampling_rate"] for x in args),
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute representations of audio from a pretrained hubert model."
    )
    parser.add_argument("model_dir", type=Path, help="Pretrained model directory")
    parser.add_argument("input_dir", type=Path, help="Root directory of input files")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("out"),
        help="Root output directory",
    )
    parser.add_argument(
        "-l",
        "--layer",
        choices=LAYERS,
        nargs="+",
        default=None,
        help="Layer(s) to extract from",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="DO NOT USE")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_dataset(
        "audiofolder",
        data_files={
            "validation": sorted(glob(str(args.input_dir / "**/*.wav"), recursive=True))
        },
        drop_labels=True,
    ).with_format("numpy")

    model = HubertForCTC.from_pretrained(args.model_dir).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_dir)

    if args.layer is None:
        layers = frozenset(
            {"z"} | {f"c{i}" for i in range(model.config.num_hidden_layers)}
        )
    elif not args.layer:
        raise ValueError("No layers selected")
    else:
        layers = frozenset(args.layer)

    model.eval()
    with torch.no_grad():
        for b, batch in tqdm(
            enumerate(
                DataLoader(
                    data["validation"],
                    batch_size=args.batch_size,
                    collate_fn=collate,
                )
            )
        ):
            assert all(
                sr == SR for sr in batch["audio"]["sampling_rate"]
            ), f"Unexpected sampling rate found in {batch['audio']['sampling_rate']}"
            input_values = feature_extractor(
                batch["audio"]["array"],
                return_tensors="pt",
                padding=True,
                sampling_rate=SR,
            ).input_values.to(device)
            if input_values.size(1) < 400: # number of samples must be >= than kernel size
                continue
            out_dict = model(input_values, output_hidden_states=True)

            for layer in layers:
                if layer == "z":
                    out = out_dict["hidden_states"][0]
                elif layer[0] == "c":
                    n = int(layer[1:])
                    out = out_dict["hidden_states"][n + 1]
                else:
                    raise ValueError(f"Invalid layer: {layer}")
                out = out[0].numpy(force=True)

                # print(out_dict["hidden_states"][int(layer[1:]) + 1][0].squeeze(1).numpy(force=True))
                # exit()

                path = Path(batch["audio"]["path"][0]).resolve()
                relpath = path.relative_to(args.input_dir.resolve())
                outpath = args.output_dir / layer / relpath.with_suffix(".npy")
                outpath.parent.mkdir(parents=True, exist_ok=True)
                np.save(outpath, out)

                if args.debug:
                    status = {
                        "path": str(outpath),
                        "audio_samples": batch["audio"]["array"][0].shape[0],
                        "input_shape": input_values.shape,
                        "num_padding": torch.sum(input_values[0] == 0.0).item(),
                        "output_shape": out.shape,
                    }
                    tqdm.write(f"{0}: {status}")

            del input_values
            del out_dict
            torch.cuda.empty_cache()
