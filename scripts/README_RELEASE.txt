Pixel Sprite Studio
===================

Text -> pixel sprite -> animation, on your own machine.

Start
-----
1. Unzip the folder anywhere you can write to (not "Program Files").
2. Run PixelSpriteStudio.exe.
   Windows SmartScreen may warn about an unsigned app: "More info" -> "Run anyway".
3. First start: the app downloads its models (about 500 MB) into the "models" folder next to
   the exe. The status bar at the bottom shows the progress. This happens once and needs an
   internet connection.
4. Type a caption, press Generate, click a sprite, pick an action on the right, press Animate.
   Or animate your own sprite: "Load PNG..." (or drop the file on the window).

Graphics card
-------------
Out of the box the app runs on any DirectX 12 graphics card (DirectML), or on the processor.

With an NVIDIA card you can make it faster: click "Download CUDA libraries" in the status bar
(about 1 GB, once), then restart the app. You need a recent NVIDIA driver; no CUDA Toolkit.

Where things are
----------------
models\           downloaded models and libraries (delete it to start over)
exports\          sprite sheets and GIFs you export
animator_ui.json  optional settings

Licences
--------
The app is MIT (LICENSE). The model weights are for NON-COMMERCIAL use (CC BY-NC 4.0,
"Built with Qwen"); see THIRD_PARTY.md.

Source, issues: https://github.com/umanets/pixelsprite-animator
