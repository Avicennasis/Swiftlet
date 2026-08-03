"""Teacher-forcing oracle for a REAL (large) mlx-lm checkpoint, bounded memory.

The 80B checkpoint cannot be resident on a 24 GB machine, so this runs the
reference forward layer by layer: for each layer it opens the model lazily
(safetensors headers only), evaluates just that layer's contribution, and
frees everything before the next one. Peak memory ~ one layer.

Dumps fixtures/real_forward.safetensors with the embedding output, the first
--capture-layers layer outputs, final-norm output, last-position logits, and
greedy ids for the Swift side (`swiftlet verify`) to reproduce.

Usage:
  .venv/bin/python scripts/gen_real_fixture.py <checkpoint-dir> [--capture-layers 4]
"""

import argparse
import gc
import json
import pathlib

import mlx.core as mx

PROMPT = "The fieldfare is a bird that"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=pathlib.Path)
    ap.add_argument("--capture-layers", type=int, default=4)
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    mx.set_default_device(mx.cpu)

    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    from mlx_lm.tokenizer_utils import load_tokenizer
    from mlx_lm.utils import load_model

    tokenizer = load_tokenizer(args.checkpoint)
    tokens = tokenizer.encode(args.prompt)
    print(f"prompt tokens ({len(tokens)}): {tokens}")
    inputs = mx.array([tokens])

    def fresh_model():
        model, _ = load_model(args.checkpoint, lazy=True)
        return model

    fixtures: dict[str, mx.array] = {"input_ids": inputs}

    model = fresh_model()
    n_layers = len(model.model.layers)
    h = model.model.embed_tokens(inputs)
    mx.eval(h)
    fixtures["embed"] = h
    fa_idx, ssm_idx = model.model.fa_idx, model.model.ssm_idx
    del model
    gc.collect()

    for i in range(n_layers):
        model = fresh_model()
        layer = model.model.layers[i]
        cache = [None] * n_layers
        fa_mask = create_attention_mask(h, cache[fa_idx])
        ssm_mask = create_ssm_mask(h, cache[ssm_idx])
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask=mask, cache=None)
        mx.eval(h)
        if i < args.capture_layers:
            fixtures[f"layer_{i:02d}"] = h
        del model, layer
        gc.collect()
        print(f"layer {i + 1}/{n_layers} done", flush=True)

    model = fresh_model()
    h = model.model.norm(h)
    logits = model.lm_head(h)
    mx.eval(logits)
    fixtures["final_norm"] = h
    fixtures["logits_last"] = logits[:, -1, :]
    greedy = mx.argmax(logits[0], axis=-1).tolist()
    del model
    gc.collect()

    out = pathlib.Path(__file__).resolve().parent.parent / "fixtures"
    mx.save_safetensors(str(out / "real_forward.safetensors"), fixtures)
    (out / "real_forward.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "prompt": args.prompt,
                "tokens": tokens,
                "greedy_next_tokens": greedy,
                "greedy_text_next": tokenizer.decode([greedy[-1]]),
                "captured_layers": args.capture_layers,
            },
            indent=2,
        )
    )
    print(f"greedy ids: {greedy}")
    print(f"next token after prompt: {tokenizer.decode([greedy[-1]])!r}")


if __name__ == "__main__":
    main()
