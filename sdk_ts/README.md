# Cutible TypeScript SDK

> TypeScript/JavaScript SDK for the Cutible agent-native montage engine.

## Installation

```bash
npm install cutible-sdk
```

## Usage

```typescript
import { CutibleClient } from 'cutible-sdk';

// Connect to a running Cutible REST API server
const client = new CutibleClient({ apiUrl: 'http://localhost:8000' });

// Create a project
await client.createProject('demo', { fps: 30, width: 1920, height: 1080 });

// Register source assets
await client.addAsset('speaker', 'video', { uri: '/path/to/speaker.mp4', duration: 60 });

// Build the edit
await client.addTrack('v_main', 'video');
await client.addClip('v_main', 'speaker', { srcIn: 0, srcOut: 10 });
await client.addTransition('clip_1', 'in', 0.5);

// Render
const result = await client.render('output.mp4', { runQc: true });
console.log(result);

// Or run the full agent swarm
const agentResult = await client.runAgent('Make a 30s recap', {
  targetDuration: 30,
  style: 'energetic',
});
```

## API Reference

### `CutibleClient`

| Method | Description |
|---|---|
| `createProject(id, opts)` | Create a new project |
| `loadProject(id, path)` | Load a project from JSON |
| `read(zoom)` | Read the timeline at a zoom level |
| `addAsset(id, type, opts)` | Register a source asset |
| `addTrack(id, kind)` | Add a track |
| `addClip(trackId, asset, opts)` | Place a clip on a track |
| `trim(clipId, opts)` | Adjust clip source in/out |
| `move(clipId, time)` | Move a clip on its track |
| `split(clipId, t)` | Split a clip at time t |
| `rippleDelete(clipId)` | Delete a clip and close the gap |
| `setSpeed(clipId, speed)` | Set clip playback speed |
| `setVolume(clipId, volume)` | Set clip volume |
| `addTransition(clipId, kind, dur)` | Add a fade in/out |
| `addTextLayer(trackId, text, in, out)` | Add a burned-in text |
| `addAudio(asset, opts)` | Add an audio clip |
| `checkpoint(label)` | Snapshot current state |
| `undo()` | Revert to last checkpoint |
| `render(output, opts)` | Render the project to video |
| `qc(file, duration)` | Run QC on a rendered file |
| `save(path)` | Save the project to JSON |
| `ingestAsset(id, uri)` | Ingest a media file |
| `searchIndex(query)` | Search the semantic index |
| `runAgent(brief, opts)` | Run the multi-agent swarm |
| `exportOtio(path)` | Export as OpenTimelineIO |
| `importOtio(path)` | Import an OpenTimelineIO file |
| `renderFarm(output, opts)` | Render using distributed farm |

## Requirements

- Node.js 18+
- A running Cutible REST API server (`cutible serve-api`)

## License

MIT
