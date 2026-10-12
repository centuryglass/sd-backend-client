# Changelog

## [0.4.0](https://github.com/centuryglass/sd-backend-client/compare/v0.3.0...v0.4.0) (2026-10-12)


### ⚠ BREAKING CHANGES

* composite ComfyUI inpainting results onto the source image ([#119](https://github.com/centuryglass/sd-backend-client/issues/119))
* apply mask_blur on ComfyUI and return the blurred mask ([#113](https://github.com/centuryglass/sd-backend-client/issues/113))

### Features

* add ParameterDef.display_name, keep description for tooltips ([#114](https://github.com/centuryglass/sd-backend-client/issues/114)) ([94155bd](https://github.com/centuryglass/sd-backend-client/commit/94155bd0b3c3db6646146504a09189494608b8b0)), closes [#109](https://github.com/centuryglass/sd-backend-client/issues/109)
* apply mask_blur on ComfyUI and return the blurred mask ([#113](https://github.com/centuryglass/sd-backend-client/issues/113)) ([9c5fbc4](https://github.com/centuryglass/sd-backend-client/commit/9c5fbc4a085425363a7dfef8b655bce35e4e2af2))
* **comfyui:** add ComfyUiWebservice.list_model_configs ([#111](https://github.com/centuryglass/sd-backend-client/issues/111)) ([73c9106](https://github.com/centuryglass/sd-backend-client/commit/73c9106c6001f608e977a342b041a91bc2cc9600))
* composite ComfyUI inpainting results onto the source image ([#119](https://github.com/centuryglass/sd-backend-client/issues/119)) ([cf8841e](https://github.com/centuryglass/sd-backend-client/commit/cf8841e47b49b73266343494eb9adb0ec23f16ca))
* support "inpaint only masked" (inpaint_full_res) on ComfyUI ([#120](https://github.com/centuryglass/sd-backend-client/issues/120)) ([c9a9aad](https://github.com/centuryglass/sd-backend-client/commit/c9a9aad5a4ebd0efb09d0789ffaf7a001614d125))


### Bug Fixes

* **comfyui:** resize in pixel space in the tiled VAE upscale fallback ([#123](https://github.com/centuryglass/sd-backend-client/issues/123)) ([529b0c8](https://github.com/centuryglass/sd-backend-client/commit/529b0c83fdc835b335c3e00c4a20bdbf31936287))

## [0.3.0](https://github.com/centuryglass/sd-backend-client/compare/v0.2.1...v0.3.0) (2026-10-09)


### ⚠ BREAKING CHANGES

* reject unknown fields in parameter models ([#93](https://github.com/centuryglass/sd-backend-client/issues/93))
* give GenerationResult the same seeds, images and control maps on both backends ([#83](https://github.com/centuryglass/sd-backend-client/issues/83))
* A1111Webservice.submit_img2img takes one DiffusionParams with the image in init_images and the mask in mask, and submit_txt2img's parameter is renamed diffusion_params.
* validate generation parameters in the pydantic models ([#77](https://github.com/centuryglass/sd-backend-client/issues/77))
* give sampler, scheduler, img2img size and denoising one meaning on both backends ([#70](https://github.com/centuryglass/sd-backend-client/issues/70))
* **comfyui:** name uploads by content hash so queued jobs keep their images ([#69](https://github.com/centuryglass/sd-backend-client/issues/69))
* add a package exception hierarchy ([#67](https://github.com/centuryglass/sd-backend-client/issues/67))

### Features

* add a Backend interface that both clients implement ([#81](https://github.com/centuryglass/sd-backend-client/issues/81)) ([108ad89](https://github.com/centuryglass/sd-backend-client/commit/108ad89652b1ebbaea8226aae4bc3f53879aa20c)), closes [#36](https://github.com/centuryglass/sd-backend-client/issues/36)
* add a package exception hierarchy ([#67](https://github.com/centuryglass/sd-backend-client/issues/67)) ([f8f8dc8](https://github.com/centuryglass/sd-backend-client/commit/f8f8dc86444798462e25122199d1d9e85286dda7))
* add command-line example scripts ([#100](https://github.com/centuryglass/sd-backend-client/issues/100)) ([f0f43ea](https://github.com/centuryglass/sd-backend-client/commit/f0f43ea12139b29a7a0a187b73be982407d20a08))
* export the ControlNet None placeholders and WebUI parameter-key constants ([#104](https://github.com/centuryglass/sd-backend-client/issues/104)) ([d3d8dfc](https://github.com/centuryglass/sd-backend-client/commit/d3d8dfcd518ce300d46f1feb1a46394e725bf9f7))
* export the supported API from the package root ([#86](https://github.com/centuryglass/sd-backend-client/issues/86)) ([ecd4f31](https://github.com/centuryglass/sd-backend-client/commit/ecd4f31f3420bbaf02d9e3d061053ee58297a1df)), closes [#24](https://github.com/centuryglass/sd-backend-client/issues/24)
* give GenerationResult the same seeds, images and control maps on both backends ([#83](https://github.com/centuryglass/sd-backend-client/issues/83)) ([8d0c125](https://github.com/centuryglass/sd-backend-client/commit/8d0c12553242f1149dd5422594bd16c64243ee61)), closes [#40](https://github.com/centuryglass/sd-backend-client/issues/40)
* give sampler, scheduler, img2img size and denoising one meaning on both backends ([#70](https://github.com/centuryglass/sd-backend-client/issues/70)) ([4c82771](https://github.com/centuryglass/sd-backend-client/commit/4c827710b0941f64d8fc888e1ad1914c18fe621d)), closes [#39](https://github.com/centuryglass/sd-backend-client/issues/39)
* list models, options and capabilities the same way on both backends ([#84](https://github.com/centuryglass/sd-backend-client/issues/84)) ([ecc3ae9](https://github.com/centuryglass/sd-backend-client/commit/ecc3ae9b7f3cd981350496bbf0da2fa0473b1d32))
* run upscaling and ControlNet preprocessor previews through generation handles ([#82](https://github.com/centuryglass/sd-backend-client/issues/82)) ([3a837ae](https://github.com/centuryglass/sd-backend-client/commit/3a837ae1ba9a0baec0ff2f2666466ad58027190a)), closes [#37](https://github.com/centuryglass/sd-backend-client/issues/37)
* validate generation parameters in the pydantic models ([#77](https://github.com/centuryglass/sd-backend-client/issues/77)) ([19db2a8](https://github.com/centuryglass/sd-backend-client/commit/19db2a870e92881adf8339aacb0c2ab68050c95e)), closes [#26](https://github.com/centuryglass/sd-backend-client/issues/26)
* **webui:** support Forge Neo and record contract fixtures for every WebUI fork ([#88](https://github.com/centuryglass/sd-backend-client/issues/88)) ([10d3bd4](https://github.com/centuryglass/sd-backend-client/commit/10d3bd4eee513ed61a16f94422f8edc13d811c54))


### Bug Fixes

* allow model fields on supported pydantic versions ([#72](https://github.com/centuryglass/sd-backend-client/issues/72)) ([55595fd](https://github.com/centuryglass/sd-backend-client/commit/55595fdcb7a26f597e4028b6e31d2a19dc06e24c)), closes [#66](https://github.com/centuryglass/sd-backend-client/issues/66)
* **comfyui:** apply CLIP skip in latent upscale workflows ([#102](https://github.com/centuryglass/sd-backend-client/issues/102)) ([1381899](https://github.com/centuryglass/sd-backend-client/commit/1381899b14eea622300cbbbec42bac685bb396ca))
* **comfyui:** apply LoRA and hypernetwork prompt tags instead of dropping them ([#76](https://github.com/centuryglass/sd-backend-client/issues/76)) ([93c4f95](https://github.com/centuryglass/sd-backend-client/commit/93c4f95735f859fa7fdc438bbc5e37472e8b8dc5)), closes [#16](https://github.com/centuryglass/sd-backend-client/issues/16)
* **comfyui:** end waits on lost jobs, report failure reasons and live progress ([#78](https://github.com/centuryglass/sd-backend-client/issues/78)) ([6698c87](https://github.com/centuryglass/sd-backend-client/commit/6698c874c4cafe7b8ce7e026f5cbbae37b776934)), closes [#19](https://github.com/centuryglass/sd-backend-client/issues/19)
* **comfyui:** name uploads by content hash so queued jobs keep their images ([#69](https://github.com/centuryglass/sd-backend-client/issues/69)) ([e4dcf67](https://github.com/centuryglass/sd-backend-client/commit/e4dcf6783128472f4efb696280cbbb00502895ce)), closes [#15](https://github.com/centuryglass/sd-backend-client/issues/15)
* **comfyui:** save generated images with a visible filename prefix ([#87](https://github.com/centuryglass/sd-backend-client/issues/87)) ([0ceea6c](https://github.com/centuryglass/sd-backend-client/commit/0ceea6c210fdb6695001e285434d933b6ddf9e32)), closes [#74](https://github.com/centuryglass/sd-backend-client/issues/74)
* **controlnet:** always send preprocessor defaults for optional parameters ([#101](https://github.com/centuryglass/sd-backend-client/issues/101)) ([d49efa0](https://github.com/centuryglass/sd-backend-client/commit/d49efa07e9539347db1527ca586bb15e912b4d77)), closes [#98](https://github.com/centuryglass/sd-backend-client/issues/98)
* reject unknown fields in parameter models ([#93](https://github.com/centuryglass/sd-backend-client/issues/93)) ([97b295c](https://github.com/centuryglass/sd-backend-client/commit/97b295c98cbd68d63c7191b09222e4c2006969c4))
* **webui:** support Forge's sampler, style and built-in ControlNet responses ([#85](https://github.com/centuryglass/sd-backend-client/issues/85)) ([be67768](https://github.com/centuryglass/sd-backend-client/commit/be67768901177be0e5aa083daf46748749f52740))


### Documentation

* add a review checklist for the integration tests' saved images ([#73](https://github.com/centuryglass/sd-backend-client/issues/73)) ([7da5d07](https://github.com/centuryglass/sd-backend-client/commit/7da5d07ba2a5a8dfe991765507faab8766756c0f))
* publish an API documentation site to GitHub Pages ([#91](https://github.com/centuryglass/sd-backend-client/issues/91)) ([5cf60de](https://github.com/centuryglass/sd-backend-client/commit/5cf60de06f94f2dfe5c53a9d4064227cac19575f))
* rewrite the README around the public API and check its examples in CI ([#89](https://github.com/centuryglass/sd-backend-client/issues/89)) ([7f7a184](https://github.com/centuryglass/sd-backend-client/commit/7f7a184dac32cb4b6105db02310ac9988edde01c))

## [0.2.1](https://github.com/centuryglass/sd-backend-client/compare/v0.2.0...v0.2.1) (2026-10-06)


### Bug Fixes

* bound HTTP timeouts, encode query params, strip trailing slashes and use wss for https ([#59](https://github.com/centuryglass/sd-backend-client/issues/59)) ([0137119](https://github.com/centuryglass/sd-backend-client/commit/0137119a2f8764c1c5e1e9ba58d3ca05e0b4de0a))
* **comfyui:** keep combo option lists on ComfyUI preprocessor parameters ([#58](https://github.com/centuryglass/sd-backend-client/issues/58)) ([dfaf613](https://github.com/centuryglass/sd-backend-client/commit/dfaf613dfa0523070c217f0fc55e02be3898f780)), closes [#34](https://github.com/centuryglass/sd-backend-client/issues/34)
* **webui:** select the checkpoint from sd_model_name ([#61](https://github.com/centuryglass/sd-backend-client/issues/61)) ([9f2585a](https://github.com/centuryglass/sd-backend-client/commit/9f2585abc88176f9b0ba32bb93dc8e7e749f3f89)), closes [#38](https://github.com/centuryglass/sd-backend-client/issues/38)

## [0.2.0](https://github.com/centuryglass/sd-backend-client/compare/v0.1.0...v0.2.0) (2026-10-06)


### ⚠ BREAKING CHANGES

* remove dead code and UI leftovers from the IntraPaint extraction ([#55](https://github.com/centuryglass/sd-backend-client/issues/55))
* rename the package to sd_backend_client ([#52](https://github.com/centuryglass/sd-backend-client/issues/52))
* **comfyui:** treat opaque/white as the region to change in inpainting masks ([#48](https://github.com/centuryglass/sd-backend-client/issues/48))

### Features

* ship a py.typed marker so type checkers read the package's annotations ([#53](https://github.com/centuryglass/sd-backend-client/issues/53)) ([f1f9533](https://github.com/centuryglass/sd-backend-client/commit/f1f95331216efe771adc399f7812fb80e913fa04)), closes [#5](https://github.com/centuryglass/sd-backend-client/issues/5)


### Bug Fixes

* **comfyui:** avoid invalid graphs for ControlNet units without image or model ([#45](https://github.com/centuryglass/sd-backend-client/issues/45)) ([56be032](https://github.com/centuryglass/sd-backend-client/commit/56be032301d0290bae656ee314de285f6a22e3d5))
* **comfyui:** make model config auto-detection robust to dot-less entries ([#44](https://github.com/centuryglass/sd-backend-client/issues/44)) ([2b7e8de](https://github.com/centuryglass/sd-backend-client/commit/2b7e8def9504c70ec8e86a1a4bfd3b9d37d36195))
* **comfyui:** resize upscale output to the requested size and apply the upscale model ([#49](https://github.com/centuryglass/sd-backend-client/issues/49)) ([b8489a9](https://github.com/centuryglass/sd-backend-client/commit/b8489a91c53b6432096736b91c3113bbb34e3605))
* **comfyui:** serialize FreeMemoryRequest in free_memory ([#43](https://github.com/centuryglass/sd-backend-client/issues/43)) ([6548742](https://github.com/centuryglass/sd-backend-client/commit/65487421a7d87364467054404461ac7e6ac4111f))
* **comfyui:** treat opaque/white as the region to change in inpainting masks ([#48](https://github.com/centuryglass/sd-backend-client/issues/48)) ([f024a59](https://github.com/centuryglass/sd-backend-client/commit/f024a591027e78acc0566564d545a2ec0a874f8e))
* stop creating directories at import time ([#42](https://github.com/centuryglass/sd-backend-client/issues/42)) ([a9feb6a](https://github.com/centuryglass/sd-backend-client/commit/a9feb6ae5e40c66a964e442b6807353d2d95f301))
* **webui:** bound the 401 retry loops and validate credentials against /sdapi ([#46](https://github.com/centuryglass/sd-backend-client/issues/46)) ([02c9bd5](https://github.com/centuryglass/sd-backend-client/commit/02c9bd50a82e6e4e1307c440164f3bb48a2acf42))
* **webui:** read ControlNet module details from the A1111 extension's module_detail key ([#56](https://github.com/centuryglass/sd-backend-client/issues/56)) ([65eae80](https://github.com/centuryglass/sd-backend-client/commit/65eae80d8613562d1be3de2ae5c787703f1cef2b)), closes [#50](https://github.com/centuryglass/sd-backend-client/issues/50)
* **webui:** send inpainting masks as opaque grayscale ([#6](https://github.com/centuryglass/sd-backend-client/issues/6)) ([a60ce40](https://github.com/centuryglass/sd-backend-client/commit/a60ce4085eef357afa85cb701ab049bbbaec3b7d))
* **webui:** stop request methods mutating the caller's DiffusionRequestBody ([#47](https://github.com/centuryglass/sd-backend-client/issues/47)) ([b91fc02](https://github.com/centuryglass/sd-backend-client/commit/b91fc022299fac3e6134270a7cb23f4f2799f4d5))


### Documentation

* add contributing guide, security policy, issue and PR templates ([#54](https://github.com/centuryglass/sd-backend-client/issues/54)) ([78df9af](https://github.com/centuryglass/sd-backend-client/commit/78df9afc06f679317675843d40a12ec0016ac409)), closes [#32](https://github.com/centuryglass/sd-backend-client/issues/32)


### Code Refactoring

* remove dead code and UI leftovers from the IntraPaint extraction ([#55](https://github.com/centuryglass/sd-backend-client/issues/55)) ([adf24ab](https://github.com/centuryglass/sd-backend-client/commit/adf24ab1ec7207acedfc3587be6399154247c966)), closes [#27](https://github.com/centuryglass/sd-backend-client/issues/27)
* rename the package to sd_backend_client ([#52](https://github.com/centuryglass/sd-backend-client/issues/52)) ([4e4ef24](https://github.com/centuryglass/sd-backend-client/commit/4e4ef242aaab84a4d725a3a24718f3ef7fd56ff5)), closes [#8](https://github.com/centuryglass/sd-backend-client/issues/8)
