# Copying an image

Many control types answer "make a new picture like this one", and they are easy to mix up. They differ in what they
take from your image and what they leave to the prompt and the checkpoint. The less a type keeps, the more freedom
the prompt has. Each row's claims come from the linked control type's page.

## What each type keeps

| Type | Keeps from your image | Leaves to the prompt |
|---|---|---|
| [[type:canny]] | Every sharp outline, as one-pixel lines: borders, folds, text, fine texture | Color, lighting, material and style |
| [[type:lineart]] | Contours and main interior lines, with most texture dropped | Color, lighting and style |
| [[type:softedge]] | Main boundaries as soft, graded lines: shapes and their rough weight | Color, material, lighting and exact outlines |
| [[type:scribble]] | Placement and general shapes of strokes, not their exact paths | What the shapes are, and every detail |
| [[type:mlsd]] | Straight line segments only | Everything curved, including people and plants |
| [[type:depth]] | Near and far: foreground, middle ground, background, rounded or flat forms | Outlines, color, texture, lighting and style |
| [[type:normal_map]] | The direction each surface faces, small bumps included | Color, material and style |
| [[type:segmentation]] | Where each kind of thing is: sky, building, person | How each thing looks inside its region |
| [[type:tile]] | Composition and colors | Fine detail, which the model redraws |
| [[type:recolor]] | Brightness: lights and shadows | Colors |
| [[type:ip2p]] | The picture, recognizably | The change the prompt asks for, read as an instruction |
| [[type:shuffle]] | Palette, textures and style | Layout and subject |
| [[type:reference]] | Subject, character or style, depending on the variant | Layout and the rest of the scene |
| [[type:ip_adapter]] | Subject, palette and style, or a face's identity, depending on the model | Exact layout, and whatever the text prompt adds |
| [[type:revision]] | The concept and mood, as an image prompt | Layout and any detail; SDXL only |

## Structure or look

The types fall into two groups, and combining one of each is common.

**Structure types keep where things are.** [[type:canny]], [[type:lineart]], [[type:softedge]], [[type:scribble]],
[[type:mlsd]], [[type:depth]], [[type:normal_map]] and [[type:segmentation]] read a map of your image's layout and
throw away its colors and surfaces. They differ in how much they pin down: Canny keeps every edge, Scribble keeps
only rough shapes, and Depth and Segmentation keep layout with no outlines at all.

**Look types keep what things look like.** [[type:ip_adapter]], [[type:reference]], [[type:shuffle]] and
[[type:revision]] carry a reference's subject, palette or style into a new picture without its layout.
[[type:recolor]] is the reverse case: it keeps the lighting and replaces only the colors.

**Tile keeps both, loosely.** [[type:tile]] reads the image itself, so composition and colors stay while detail is
redrawn. That makes it the type for upscaling and for variations that stay close to the original.

The [[type:shuffle]] page gives the classic pairing: a Canny unit on one image for the structure, and a Shuffle unit
on another for the style. A Depth, Canny or OpenPose unit with an IP-Adapter unit works the same way, per the
[[type:ip_adapter]] page.

## Picking one

- **Same picture, new style or material:** Canny for photos and renders with crisp edges, Lineart for a drawn
  look, Soft Edge when outlines should loosen.
- **Same scene, different things in it:** Depth or Segmentation, which keep the layout without pinning outlines.
- **Same picture, more detail or a larger size:** Tile.
- **Same subject or style, new picture:** IP-Adapter, Reference, Shuffle or Revision.
- **Same picture, one change described in words:** Instruct Pix2Pix, on SD 1.5.
