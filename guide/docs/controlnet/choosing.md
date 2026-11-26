# Which one do I use?

Start from what you want to keep from your image. Each answer links to a control type, whose page says which
preprocessor and model to pick. Several units can run together, such as one for the layout and one for the style.

## Keep the composition

| You want to keep | Use | Notes |
|---|---|---|
| Every outline, exactly where it is | [[type:canny]] | One-pixel edges. Fine texture shows up as lines too. |
| The main lines, as an illustrator would ink them | [[type:lineart]] | Pick the preprocessor by source: photo, anime or an existing drawing. |
| The shapes, but not every fine line | [[type:softedge]] | Soft, graded boundaries leave room for new color and material. |
| A rough layout from a doodle or a photo | [[type:scribble]] | Keeps placement and general shapes, not exact strokes. |
| Walls, furniture and buildings | [[type:mlsd]] | Straight lines only; curves and people drop out. |
| What is near and what is far | [[type:depth]] | Ignores outlines and surface detail. |
| The 3D shape of surfaces, bumps included | [[type:normal_map]] | Keeps surface angles a depth map flattens. |
| Where sky, road, trees and people are | [[type:segmentation]] | Flat color regions, one color per kind of thing. Can be painted by hand. |

## Keep a pose

| You want to keep | Use | Notes |
|---|---|---|
| The pose of one or more people | [[type:openpose]] | A stick figure per person. DWPose, animal pose and face landmarks are here too. |
| Where a face sits and which way it turns | [[type:instant_id]] | Its ControlNet unit reads face keypoints. SDXL only. |
| The general layout and pose, not the content | [[type:ip_adapter]] | The IP Composition Adapters. |

## Keep a style, a subject or a palette

| You want to keep | Use | Notes |
|---|---|---|
| A reference's subject and look, without its layout | [[type:ip_adapter]] | Reads the image as a prompt. Works alongside a text prompt. |
| A subject or style, with no model to download | [[type:reference]] | Reference Only for the subject, Reference AdaIN for the style. |
| A painting's palette, textures and brushwork | [[type:shuffle]] | SD 1.5 only. Pair with a structure unit such as Canny. |
| The concept and mood of an image, as a prompt | [[type:revision]] | SDXL only, no model to download. |
| A color scheme laid out in blocks | [[type:t2i_adapter]] | The color adapter, fed by Color Grid. |
| The lights and shadows, with new colors | [[type:recolor]] | Colorizes black-and-white photos. |

## Keep a face

| You want to keep | Use | Notes |
|---|---|---|
| A person's identity, in a new style or scene | [[type:ip_adapter]] | The FaceID, FaceID Plus and PuLID models. |
| Identity and face position, from one photo | [[type:instant_id]] | Two units in order. SDXL only. |
| Identity tied to a word in the prompt | [[type:photomaker]] | reForge only as a control type; ComfyUI has core nodes. SDXL only. |
| A face's look, from a cropped photo | [[type:ip_adapter]] | Plus Face and Full Face need only a cropped face photo. |

## Copy this image closely

Several types can make a new image that looks like a given one, and they differ in what they keep and what they
leave to the prompt. [Copying an image](copying-an-image.md) compares them side by side. In short:

- To keep the picture and redraw its detail, use [[type:tile]].
- To keep the picture and apply an instruction ("make it winter"), use [[type:ip2p]] on SD 1.5.
- To keep the outlines and change everything else, use [[type:canny]] or [[type:lineart]].

## Upscale

Use [[type:tile]] with a tiled upscaler such as Ultimate SD Upscale. Each tile follows its own content instead of the
whole prompt, so faces don't appear in leaves. Tile also repairs blurry or badly upscaled images.

## Inpaint and outpaint

Use [[type:inpaint]] to repaint a masked area with any checkpoint, without a merged inpainting checkpoint. To extend
the canvas, mask the new border and inpaint it. Inpaint Only + LaMa suits object removal and outpainting.

## Hide a QR code, text or a shape

Use [[type:qr_code]]. The QR and illusion models take the code or shape directly, with no preprocessor.

## Animate

Use [[type:sparsectrl]] with AnimateDiff on SD 1.5 to guide an animation from one or a few images or sketches.
