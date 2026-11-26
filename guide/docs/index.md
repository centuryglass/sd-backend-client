# Stable Diffusion guide

A guide for artists to the tools Stable Diffusion front ends put in front of you: what each one does, what it is
good for, and which to pick. It covers the A1111 WebUI, the Forge family (Forge, reForge and Forge Neo) and ComfyUI,
and gives the name each of them uses, so you can find a tool in your own dropdown.

## Sections

- **[ControlNet](controlnet/index.md):** guiding a generation with an image, such as an edge map, a depth map, a
  pose or a face. Start with [Which one do I use?](controlnet/choosing.md) if you know what you want to keep from
  your image, or [Name lookup](controlnet/names.md) if you have a name from your backend and want to know what it
  is.

## Where this comes from

Every page about a specific control type, preprocessor or model is generated from a sourced record in the
[`data/controlnet/`](https://github.com/centuryglass/sd-backend-client/tree/main/data/controlnet) folder of the
sd-backend-client repository. Each record page ends with its sources, and [Sources](controlnet/sources.md) lists
them all. A page marked "Drafted record" still has open questions, listed on the page.

This guide is part of [sd-backend-client](https://github.com/centuryglass/sd-backend-client), a Python library for
driving these backends. Programmers looking for that library want the
[API reference](https://centuryglass.github.io/sd-backend-client/).
