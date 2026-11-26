# ControlNet

ControlNet lets you guide a generation with an image as well as a prompt. You give it a **control image**, such as
an outline, a depth map or a pose skeleton, and the generated picture follows it: the same edges, the same layout in
depth, the same pose. What the control image leaves out, the prompt and the checkpoint fill in.

## Three parts: control type, preprocessor, model

Every ControlNet unit in the WebUI, Forge or a ComfyUI workflow combines three choices.

**The control type** is the kind of guidance: what the unit keeps from your image. [[type:canny]] keeps sharp
outlines, [[type:depth]] keeps what is near and what is far, [[type:openpose]] keeps the pose of each person. The
WebUI and Forge show it as a row of buttons that filters the other two lists. ComfyUI has no control types; you
pick a preprocessor node and a model directly.

**The preprocessor** turns your picture into the control image that a model reads. The Canny preprocessor turns a
photo into thin white lines on black; a depth estimator turns it into a grayscale map, white near and black far. If
your image already is a control image, such as a pose skeleton you posed in an editor or a QR code, use no
preprocessor. Each preprocessor's page shows what it outputs, what input it handles well, and its settings.

**The model** is the trained network that reads the control image during generation and steers each step toward
it. A model is trained on one kind of map, so it needs a preprocessor whose output matches. Each control type's page
lists its preprocessors and models in recommended order, and says why.

A few types need no model file, such as [[type:reference]], which feeds the reference image through the checkpoint
you are already using.

## SD 1.5 and SDXL

**A ControlNet model works only with checkpoints of the base model it was trained for.** An SD 1.5 model does
nothing useful with an SDXL checkpoint, and the reverse. Every model page names its base model, and the model tables
mark it in the **Base** column. Fine-tuned checkpoints, such as anime or photographic merges, share their base
model's ControlNets.

Preprocessors don't depend on the base model. A Canny map works with SD 1.5 and SDXL Canny models alike.

Some control types exist for one base model only: [[type:instant_id]], [[type:photomaker]] and [[type:revision]]
are SDXL only, and [[type:shuffle]], [[type:ip2p]] and [[type:sparsectrl]] are SD 1.5 only.

## Kinds of model

Model pages also give the model's kind, which affects size and speed more than results:

- **ControlNet:** a trainable copy of part of Stable Diffusion that reads the control image. The original design,
  and the largest files.
- **T2I-Adapter**, **Control-LoRA** and **ControlNet-LLLite:** smaller networks that do the same job at lower
  memory cost. See [[type:t2i_adapter]] for TencentARC's adapters.
- **Union ControlNet:** one model that covers several control types, such as
  [[model:xinsir_union_promax_sdxl]].
- **IP-Adapter:** reads a reference image as a prompt rather than as a map. See [[type:ip_adapter]].

## Where to go next

- [Which one do I use?](choosing.md) starts from what you want to keep.
- [Copying an image](copying-an-image.md) compares the many ways of saying "make it look like this image".
- [Control types](types/index.md), [Preprocessors](preprocessors/index.md) and [Models](models/index.md) list
  every record.
- [Name lookup](names.md) finds a name from your backend's dropdown, such as
  `control_v11p_sd15_canny [d14c016b]`.
