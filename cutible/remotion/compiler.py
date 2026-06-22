"""Remotion project compiler.

Converts Timeline-as-Data into Remotion (React) source code for
programmatic rendering. Handles video clips, audio tracks, and text overlays.
"""

from __future__ import annotations

import json
import os
import shutil

from ..schema import Project, TrackKind


class RemotionCompiler:
    """Compile Timeline-as-Data into a Remotion project.

    Generates a self-contained Remotion project with:
    - Root composition from the timeline
    - Video clip components (<Video>)
    - Audio components (<Audio>)
    - Text layer components (titles, lower thirds, captions)
    - Config for deterministic rendering
    """

    def __init__(self, project: Project):
        self.p = project

    def generate_project(self, output_dir: str) -> dict:
        """Generate a complete Remotion project directory."""
        os.makedirs(output_dir, exist_ok=True)
        pkg = {
            "name": f"cutible-{self.p.id}",
            "version": "1.0.0",
            "private": True,
            "scripts": {
                "start": "remotion studio",
                "build": "remotion render src/index.ts CutibleVideo out/video.mp4",
                "render": "remotion render src/index.ts CutibleVideo",
            },
            "dependencies": {
                "react": "^18.0.0",
                "react-dom": "^18.0.0",
                "remotion": "^4.0.0",
                "@remotion/cli": "^4.0.0",
            },
        }
        with open(os.path.join(output_dir, "package.json"), "w") as f:
            json.dump(pkg, f, indent=2)
        tsconfig = {
            "compilerOptions": {
                "target": "ES2020",
                "module": "commonjs",
                "jsx": "react-jsx",
                "strict": True,
                "esModuleInterop": True,
                "skipLibCheck": True,
                "forceConsistentCasingInFileNames": True,
                "outDir": "./dist",
            },
            "include": ["src/**/*"],
        }
        with open(os.path.join(output_dir, "tsconfig.json"), "w") as f:
            json.dump(tsconfig, f, indent=2)
        src_dir = os.path.join(output_dir, "src")
        os.makedirs(src_dir, exist_ok=True)
        self._write_root(src_dir)
        self._write_composition(src_dir)
        self._write_video_clip(src_dir)
        self._write_text_layers(src_dir)
        self._write_config(src_dir)
        media_dir = os.path.join(output_dir, "public", "media")
        os.makedirs(media_dir, exist_ok=True)
        self._copy_media(media_dir)
        return {"ok": True, "project_dir": output_dir, "files": self._list_files(output_dir)}

    def _copy_media(self, media_dir: str) -> list[str]:
        """Copy referenced media files into the Remotion project public dir."""
        copied = []
        for asset in self.p.assets:
            if asset.uri and os.path.exists(asset.uri):
                ext = os.path.splitext(asset.uri)[1]
                dest = os.path.join(media_dir, f"{asset.id}{ext}")
                try:
                    shutil.copy2(asset.uri, dest)
                    copied.append(dest)
                except Exception:
                    pass
        return copied

    def render_to_frames(self, output_dir: str, fps: int = 30) -> dict:
        """Generate Remotion project configured for frame-by-frame rendering."""
        result = self.generate_project(output_dir)
        config_path = os.path.join(output_dir, "src", "config.ts")
        with open(config_path) as f:
            content = f.read()
        content = content.replace(
            "IMAGE_FORMAT: 'jpeg'",
            f"IMAGE_FORMAT: 'png'\n  FPS: {fps}",
        )
        with open(config_path, "w") as f:
            f.write(content)
        return result

    def _write_root(self, src_dir: str) -> None:
        root_ts = 'import { Composition } from "remotion";\n'
        root_ts += 'import { CutibleComposition } from "./Composition";\n\n'
        root_ts += "export const RemotionRoot: React.FC = () => {\n"
        root_ts += "  return (\n"
        root_ts += "    <>\n"
        root_ts += "      <Composition\n"
        root_ts += '        id="CutibleVideo"\n'
        root_ts += "        component={CutibleComposition}\n"
        root_ts += f"        durationInFrames={{{int(self.p.duration * self.p.fps)}}}\n"
        root_ts += f"        fps={{{self.p.fps}}}\n"
        root_ts += f"        width={{{self.p.width}}}\n"
        root_ts += f"        height={{{self.p.height}}}\n"
        root_ts += "      />\n"
        root_ts += "    </>\n"
        root_ts += "  );\n"
        root_ts += "};\n"
        with open(os.path.join(src_dir, "Root.tsx"), "w") as f:
            f.write(root_ts)
        index_ts = 'import { registerRoot } from "remotion";\n'
        index_ts += 'import { RemotionRoot } from "./Root";\n\n'
        index_ts += "registerRoot(RemotionRoot);\n"
        with open(os.path.join(src_dir, "index.ts"), "w") as f:
            f.write(index_ts)

    def _write_composition(self, src_dir: str) -> None:
        clips_by_track: dict[str, list] = {}
        texts_by_track: dict[str, list] = {}
        for track in self.p.tracks:
            if track.kind == TrackKind.video:
                clips_by_track[track.id] = list(track.clips)
                texts_by_track[track.id] = list(track.texts)
            elif track.kind == TrackKind.audio:
                clips_by_track[track.id] = list(track.clips)

        comp_ts = 'import { AbsoluteFill, Sequence, Audio, staticFile } from "remotion";\n'
        comp_ts += 'import { VideoClip } from "./VideoClip";\n'
        comp_ts += 'import { TextLayer } from "./TextLayer";\n\n'
        comp_ts += "export const CutibleComposition: React.FC = () => {\n"
        comp_ts += f"  const fps = {self.p.fps};\n"
        comp_ts += "  return (\n"
        bg = self.p.globals.background
        comp_ts += '    <AbsoluteFill style={{"backgroundColor": "' + bg + '"}}>\n'

        for track_id, clips in clips_by_track.items():
            if track_id.startswith("v") or track_id == "v_main":
                for clip in clips:
                    start_frame = int(clip.timeline_in * self.p.fps)
                    dur_frames = int(clip.src_duration * self.p.fps)
                    asset = self.p.asset(clip.asset)
                    media_file = f"media/{clip.asset}{self._get_ext(asset.uri)}"
                    comp_ts += f"      <Sequence from={{{start_frame}}} durationInFrames={{{dur_frames}}}>\n"
                    comp_ts += "        <VideoClip\n"
                    comp_ts += f'          src={{staticFile("{media_file}")}}\n'
                    comp_ts += f"          srcStart={{{clip.src_in}}}\n"
                    comp_ts += f"          srcEnd={{{clip.src_out}}}\n"
                    comp_ts += f"          volume={{{clip.volume}}}\n"
                    comp_ts += f"          speed={{{clip.speed}}}\n"
                    comp_ts += "        />\n"
                    comp_ts += "      </Sequence>\n"
            elif track_id.startswith("a") or track_id == "music":
                for clip in clips:
                    start_frame = int(clip.timeline_in * self.p.fps)
                    dur_frames = int(clip.src_duration * self.p.fps)
                    asset = self.p.asset(clip.asset)
                    media_file = f"media/{clip.asset}{self._get_ext(asset.uri)}"
                    comp_ts += f"      <Sequence from={{{start_frame}}} durationInFrames={{{dur_frames}}}>\n"
                    comp_ts += "        <Audio\n"
                    comp_ts += f'          src={{staticFile("{media_file}")}}\n'
                    comp_ts += f"          volume={{{clip.volume}}}\n"
                    comp_ts += "        />\n"
                    comp_ts += "      </Sequence>\n"

        for _track_id, texts in texts_by_track.items():
            for text in texts:
                start_frame = int(text.timeline_in * self.p.fps)
                dur_frames = int(text.duration * self.p.fps)
                comp_ts += (
                    f"      <Sequence from={{{start_frame}}} durationInFrames={{{dur_frames}}}>\n"
                )
                comp_ts += "        <TextLayer\n"
                comp_ts += f'          text="{text.text}"\n'
                comp_ts += f"          fontSize={{{text.font_size}}}\n"
                comp_ts += f'          fontColor="{text.font_color}"\n'
                comp_ts += f'          x="{text.x}"\n'
                comp_ts += f'          y="{text.y}"\n'
                comp_ts += "        />\n"
                comp_ts += "      </Sequence>\n"

        comp_ts += "    </AbsoluteFill>\n"
        comp_ts += "  );\n"
        comp_ts += "};\n"
        with open(os.path.join(src_dir, "Composition.tsx"), "w") as f:
            f.write(comp_ts)

    def _write_video_clip(self, src_dir: str) -> None:
        video_ts = """import React from "react";
import { Video, Img, staticFile } from "remotion";

interface VideoClipProps {
  src: string;
  srcStart?: number;
  srcEnd?: number;
  volume?: number;
  speed?: number;
}

export const VideoClip: React.FC<VideoClipProps> = ({
  src,
  srcStart = 0,
  srcEnd,
  volume = 1.0,
  speed = 1.0,
}) => {
  const startFrom = Math.floor(srcStart * 30);
  const endAt = srcEnd ? Math.floor(srcEnd * 30) : undefined;

  return (
    <Video
      src={src}
      startFrom={startFrom}
      endAt={endAt}
      volume={volume}
      playbackRate={speed}
      style={{
        width: "100%",
        height: "100%",
        objectFit: "cover",
      }}
    />
  );
};
"""
        with open(os.path.join(src_dir, "VideoClip.tsx"), "w") as f:
            f.write(video_ts)

    def _write_text_layers(self, src_dir: str) -> None:
        text_ts = """import React from "react";

interface TextLayerProps {
  text: string;
  fontSize?: number;
  fontColor?: string;
  x?: string;
  y?: string;
}

export const TextLayer: React.FC<TextLayerProps> = ({
  text,
  fontSize = 48,
  fontColor = "white",
  x = "50%",
  y = "90%",
}) => {
  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y,
        transform: "translate(-50%, -50%)",
        fontSize: `${fontSize}px`,
        color: fontColor,
        fontFamily: "Arial, sans-serif",
        fontWeight: "bold",
        textShadow: "2px 2px 4px rgba(0,0,0,0.8)",
        padding: "8px 16px",
        backgroundColor: "rgba(0,0,0,0.5)",
        borderRadius: "4px",
      }}
    >
      {text}
    </div>
  );
};
"""
        with open(os.path.join(src_dir, "TextLayer.tsx"), "w") as f:
            f.write(text_ts)

    def _write_config(self, src_dir: str) -> None:
        config_ts = f"""export const VIDEO_CONFIG = {{
  WIDTH: {self.p.width},
  HEIGHT: {self.p.height},
  FPS: {self.p.fps},
  DURATION_FRAMES: {int(self.p.duration * self.p.fps)},
  IMAGE_FORMAT: 'jpeg',
  QUALITY: 80,
}};
"""
        with open(os.path.join(src_dir, "config.ts"), "w") as f:
            f.write(config_ts)

    def _get_ext(self, uri: str) -> str:
        if not uri:
            return ".mp4"
        ext = os.path.splitext(uri)[1]
        return ext if ext else ".mp4"

    def _list_files(self, directory: str) -> list[str]:
        files = []
        for root, _, filenames in os.walk(directory):
            for fn in filenames:
                rel = os.path.relpath(os.path.join(root, fn), directory)
                files.append(rel)
        return files
