# Third-party notices

The files below are retained only to instantiate the frozen Mammo-CLIP
EfficientNet-B2 image encoder used by the manuscript:

- `src/dagl_mil/models/mammoclip/efficientnet_custom.py`
- `src/dagl_mil/models/mammoclip/efficient_net_custom_utils.py`

They originate from the EfficientNet-PyTorch implementation by Luke Melas and
the image-encoder implementation distributed by the official
[Mammo-CLIP repository](https://github.com/batmanlab/Mammo-CLIP). Original
author notices remain in the source files.

The complete Mammo-CLIP license text is retained with this deliverable as
`MAMMO_CLIP_LICENSE_NOTICE.txt` (Creative Commons BY-NC-SA 4.0). Its line
content is identical to the upstream license; relocation through the patch
tool normalized one line-ending byte. The retained file has SHA-256
`E4963935F432D69D59EE24433336DE3AB3AB056CBF476B4A2D309E59E4D9EA7F`.

The pretrained checkpoint is an external artifact and is not committed to this
repository. The formal experiments used a repacked, weights-only copy with
SHA-256 `F0559EE74E83BB6930392056183AF3ECBD72D262C469501A0F982C66A913339E`.

The paper calls this checkpoint `b2-model-best-epoch-10.tar`; the repacked file
contains the same extracted PyTorch archive available on this workstation.
