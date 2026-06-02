"""Idempotently add the --surgenet_weights hook to a fresh Surgical-Mamba train.py
(so a SurgeNet/surgical-pretrained ConvNeXt-V2 can init the backbone)."""
import sys

path = sys.argv[1]
src = open(path).read()
if "surgenet_weights" in src:
    print("patch already applied"); sys.exit(0)

arg_anchor = '    p.add_argument("--backbone", default="convnext_tiny")'
arg_add = arg_anchor + '\n    p.add_argument("--surgenet_weights", default=None,\n                   help="SurgeNet ConvNeXt-V2 checkpoint to init the backbone")'

load_anchor = "    model = CausalSurgicalMamba.from_config(cfg).to(device)"
load_add = load_anchor + '''

    if getattr(args, "surgenet_weights", None):
        from timm.models.convnext import checkpoint_filter_fn
        _ck = torch.load(args.surgenet_weights, map_location="cpu", weights_only=False)
        _sd = _ck.get("teacher", _ck.get("model", _ck)) if isinstance(_ck, dict) else _ck
        _sd = {k: v for k, v in _sd.items() if hasattr(v, "shape")}
        _conv = checkpoint_filter_fn(_sd, model.extractor.backbone)
        _m, _u = model.extractor.backbone.load_state_dict(_conv, strict=False)
        print(f"[SurgeNet] backbone init from {args.surgenet_weights} (missing={len(_m)} unexpected={len(_u)})")'''

if arg_anchor not in src or load_anchor not in src:
    print("ERROR: anchors not found — train.py layout changed; patch manually"); sys.exit(1)

src = src.replace(arg_anchor, arg_add, 1).replace(load_anchor, load_add, 1)
open(path, "w").write(src)
print("patched train.py: added --surgenet_weights hook")
