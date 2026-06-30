# SceneRecon2 Editor Controls

This document is the source of truth for the browser editor shortcuts. When changing keyboard behavior in `SceneRecon2/place_empty_cup_editor.py`, update this file in the same change.

## Core Editing

- `1`-`9`: select object by the number shown in the object panel.
- `Tab`: cycle through editable objects.
- `C`: select the camera.
- `Arrow keys` or `W/A/S/D`: move the selected object on the table. When the camera is selected, these move the camera in its editable y/z plane.
- Hold `Z`: fine adjustment mode.
- Hold `X`: coarse adjustment mode.
- No modifier: normal adjustment mode.
- `Q` / `E`: rotate selected object yaw. When the camera is selected, these adjust camera pitch.
- `,` / `.`: switch selected object's model to previous/next candidate.

## Review And Navigation

- Hold `Space`: temporarily show the target first frame in the current render pane.
- Release `Space`: return to the current render.
- `Enter`: save reconstruction.
- `+`: switch to the next unfinished episode in the current task.
- `-`: switch to the previous unfinished episode in the current task.

## Reserved

- Roll, pitch, and scale controls for individual objects are not implemented yet.
- Future shortcut changes should avoid browser-global shortcuts and should be mirrored here.
