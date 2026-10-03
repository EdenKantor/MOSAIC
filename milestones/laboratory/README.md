# MOSAIC Laboratory

A local, read-only research viewer for saved MOSAIC event traces. Replay actor observations, advertised actions, proposed and accepted/rejected actions, runtime role, outcomes, calls, token dimensions, latency and escalations. Inspect the same task and seed side by side, with step synchronization or event-index synchronization.

**ENGINEERING VIEWER.** The viewer makes recorded behavior inspectable. It does not establish Weak/Teacher superiority, learning, transfer, generalization, retention or cost savings. Scripted-provider artifacts are labelled as fixtures. M3 remains stopped.

**Runtime role** comes from public event/accounting metadata. During candidate diagnostics, a recorded `weak` or `teacher` annotation does not identify an empirically selected Weak/Teacher hierarchy. Models remain candidates until matched task evidence supports role selection; the viewer does not infer roles from model names, sizes or one-action smoke results.

## Launch

Python 3.12 or later is sufficient; no additional packages, provider credentials, Alem installation or model service are needed. From the repository root:

```text
cd milestones/laboratory
python -m laboratory --run-root ../../runs
```

Open [MOSAIC Laboratory](http://127.0.0.1:8765). Stop the server with Ctrl+C. The server binds only to `127.0.0.1`; it has no option to expose a public interface. A different local port can be selected with `--port 8766`.

Whitelist additional existing artifact directories at launch:

```text
python -m laboratory --run-root ../../runs --run-root ../milestone-1.25/outputs
```

Each directory containing `events.jsonl` becomes a selectable episode. `summary.json` and calibration `arm.json` are optional. Run IDs are opaque; the browser cannot request arbitrary filesystem paths. Discovery happens at server startup. Reload refreshes known artifacts; restart to discover newly created episode directories.

Use Play/Pause, previous/next, speed, the slider or timeline rows to navigate a completed trace. The visible-world pane displays **only recorded actor text**; inventory/status appear when they are present in that text. The viewer never reconstructs a world from private snapshots. Event inspection and final summary disclosure show explicit public projections rather than raw artifacts.

Comparison choices require identical recorded task and seed. Step synchronization aligns the step number and ordinal event within that step; event types and call counts can differ. A shorter trace remains at its final applicable event. These controls are replay navigation, not environment actions. Matching task/seed labels do not prove identical initial conditions, prompts, budgets or scientific comparability.

## Public and privileged information

The default API/UI omits full configurations, manifests, simulator snapshots, PRNG state, full achievements, verifier evidence, unknown event payloads, model history/provenance fields and arbitrary exception text. Outcome booleans and final summaries are explicitly labelled research metadata; they are not passed to any actor. Optional usage dimensions missing from older traces remain unknown. The replay reports known subtotals and unknown-call counts, does not add reasoning/cache subset counts to other dimensions, and does not claim cross-provider token comparability.

Privileged replay evidence requires **both** an operator launch flag and the browser's explicit audit checkbox:

```text
python -m laboratory --run-root ../../runs --allow-audit
```

The separate **AUDIT VIEW** endpoint serves only recorded simulator snapshots and verifier evidence; it still omits configs/manifests. Audit opt-in displays the most recent privileged primary-run event at or before the replay position. Turning it off clears the audit payload from the page. Audit access is a local inspection boundary, not authentication or a security sandbox: anyone who can access the local server can intentionally opt in when its operator enables it. It must never be used to provide actor inputs.

The viewer neither reads provider credential variables nor makes provider/network requests. Known credential string patterns and credential-named audit fields are redacted; arbitrary exception messages are withheld. Use controlled MOSAIC artifacts: textual observations/actions are displayed as recorded public data, and this reader cannot identify every possible secret embedded inside arbitrary prose. It creates no credential files and does not serialize raw configs.

## Saved official Alem frames

Current text-only MOSAIC traces need no frame files. If an experiment already saved actor-visible PNG/JPEG output from the official Alem renderer, an operator can provide a `viewer-frames.json` beside that episode's `events.jsonl`:

```json
{
  "schema_version": 1,
  "frames": [
    {
      "sequence": 2,
      "path": "frames/actor-0002.png",
      "source": "alem_official_renderer",
      "view": "actor_visible"
    }
  ]
}
```

The sequence must exist in the trace. Paths must be relative to that episode; traversal, absolute paths, links, non-PNG/JPEG content, full-world frames and other renderer declarations are rejected. The viewer trusts the artifact author's renderer/visibility declaration; it cannot establish image provenance from pixels. It does not render from a private snapshot, create images, import Alem or change recorder behavior. Full-world imagery is never served through this frame endpoint.

## Read-only API

| GET route | Response |
| --- | --- |
| `/api/runs` | Projected metadata for discovered episodes and audit availability |
| `/api/runs/<opaque-id>` | Projected public events, summary, warnings and saved-frame URLs |
| `/api/runs/<opaque-id>/audit?ack=privileged-replay` | Privileged evidence, only with server audit opt-in |
| `/api/runs/<opaque-id>/frames/<sequence>` | An explicitly declared actor-visible official frame |

POST/PUT/PATCH/DELETE/HEAD/OPTIONS are rejected. There is no runtime-control, inference, streaming or game endpoint. Static assets use no external fonts/scripts/CDNs; artifact strings enter the page through `textContent`, not HTML. Host/origin checks, no CORS, restrictive browser content policy and `no-store` responses reduce accidental cross-origin exposure. Artifact roots are whitelisted at startup; resolved paths must stay under those roots, and symbolic links/junctions are rejected. Malformed or partial JSONL retains earlier complete events with a visible warning.

Reader bounds: 1,000 episodes; discovery depth 12; 128 MiB per JSONL; 32 MiB per line/JSON companion; 100,000 events per trace; 16 MiB per frame. Larger datasets should be separated into curated roots. The viewer assumes trusted local artifact storage; it is not intended to defend against another process racing filesystem mutations. It supports schema 1–3 core event names and the current calibration episode layout. Unknown future payloads are suppressed until explicitly supported.

## Verification

The test suite uses temporary synthetic artifacts and a local HTTP server. It checks nested private-field exclusion, separate audit opt-in, missing usage, credential redaction without environment reads, path/root confinement, linked artifacts, image visibility/provenance/path checks, partial traces, size bounds, HTTP methods, cross-origin rejection and unchanged artifact bytes.

```text
python -m unittest discover -s tests -v
```

When developer check tools are already available:

```text
python -m ruff format --check laboratory tests
python -m ruff check laboratory tests
python -m mypy laboratory tests
```

Test results and browser verification are recorded only after execution. No scientific or real-model results are generated by these checks.

When Node.js is already available, the optional replay calculation check uses only its standard library:

```text
node --check laboratory/static/app.js
node tests/replay_logic.cjs
```

This checks step synchronization, visible-state isolation, failed-call accounting, missing usage and rejection feedback without a browser or network. It is not a visual layout test.

Local verification on 2026-10-03: **16 Python tests passed**; format/lint passed; strict mypy passed for 6 Python files; JavaScript syntax and replay calculation checks passed. The three existing repository `runs` traces (two fake pilots and one Alem environment smoke using a scripted provider) loaded through the public projection without warnings or snapshots; all have zero saved frames. No inference was performed. Browser automation reported no available browsers in this session, so visual layout and interactive browser behavior remain unverified.
