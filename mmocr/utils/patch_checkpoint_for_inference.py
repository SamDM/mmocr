from typing import Optional
import os.path as osp

import torch
from mmcv.runner import _load_checkpoint


def patch_checkpoint_for_inference(inp_filename: str, out_filename: Optional[str] = None):

    if out_filename is None:
        no_ext, ext = osp.splitext(inp_filename)
        out_filename = no_ext + "_patched" + ext

    if not osp.isfile(inp_filename):
        raise FileNotFoundError(inp_filename)

    if osp.isfile(out_filename):
        print(f"skipping input:\n\t{inp_filename}\nbecause output exists:\n\t{out_filename}")
        return

    cpt = _load_checkpoint(inp_filename)
    if 'meta' not in cpt.keys():
        cpt['meta'] = dict(CLASSES='text')

    with open(out_filename, 'wb') as f:
        torch.save(cpt, f)
        f.flush()


if __name__ == '__main__':
    patch_checkpoint_for_inference("checkpoints/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth")
