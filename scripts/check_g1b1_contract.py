"""Check one manually annotated, real vision_trial.capture frame against the Skill.

This diagnostic never runs Isaac Sim, changes an observation, or commits a skill.
It requires the original run's protocol, observation, frame index and RGB PNG.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("/tmp/quadrupedarena_g1b1_contract_check")
FORMAL_ENGINE = "Python kinematics/referee; Isaac online RGB only; no PhysX steps"


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def png_size(path):
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a PNG with an IHDR header: {path}")
    return struct.unpack(">II", header[16:24])


def output_path(directory):
    path = directory.expanduser().resolve()
    temporary_roots = (Path("/tmp").resolve(), Path(tempfile.gettempdir()).resolve())
    if not any(path == root or root in path.parents for root in temporary_roots):
        raise ValueError("Output directory must be inside a system temporary directory")
    path.mkdir(parents=True, exist_ok=True)
    return path / "contract_check.json"


def check(args):
    run = args.run_dir.expanduser().resolve()
    record = {
        "status": "failed", "input_origin": "unverified",
        "same_frame_verified": False, "validation_passed": False,
        "projection_passed": False,
        "input": {"run_dir": str(run), "observation_id": args.observation_id,
                  "frame_file": args.frame_file},
        "manual_annotation": {"feet": list(args.feet), "note": args.annotation_note,
                              "review": "user_supplied_not_machine_verified"},
        "sighting": None, "world_xy_m": None, "error": None,
    }
    try:
        if not args.annotation_note.strip():
            raise ValueError("Manual annotation note must describe a visible ground-contact foot")
        protocol = read_json(run / "protocol.json")
        if (protocol.get("engine") != FORMAL_ENGINE
                or "skill_api_version" not in protocol):
            raise ValueError("Run protocol is not the formal RGB semantic Skill trial")
        observation_path = run / "observations" / f"{args.observation_id:03d}.json"
        packet = read_json(observation_path)
        if type(packet.get("observation_id")) is not int or packet["observation_id"] != args.observation_id:
            raise ValueError("Observation ID does not match the observation file")
        frames = packet.get("frames")
        if not isinstance(frames, list):
            raise ValueError("Observation has no frames list")
        selected = [frame for frame in frames if frame.get("file") == args.frame_file]
        if len(selected) != 1:
            raise ValueError("Selected frame must occur exactly once in the observation")
        frame = selected[0]
        index = read_json(run / "frame_index.json")
        if not isinstance(index, list) or sum(item == frame for item in index) != 1:
            raise ValueError("Selected frame metadata does not match one capture index entry")
        relative_frame = Path(args.frame_file)
        frame_path = (run / relative_frame).resolve()
        if (relative_frame.suffix.lower() != ".png" or len(relative_frame.parts) != 2
                or relative_frame.parts[0] != "frames"
                or frame_path.parent != (run / "frames").resolve()):
            raise ValueError("Frame path must identify a PNG directly inside run/frames")
        if png_size(frame_path) != (768, 432):
            raise ValueError("Captured PNG is not 768 by 432 pixels")
        capture_time = frame["simulation_seconds"]
        if type(capture_time) not in (int, float) or not math.isfinite(capture_time):
            raise ValueError("Selected frame has no finite simulation_seconds")
        matching_times = [item for item in frames if item.get("simulation_seconds") == capture_time]
        if matching_times != [frame]:
            raise ValueError("Selected frame time is absent or ambiguous in the observation")

        record["input_origin"] = "formal_vision_trial_capture"
        record["same_frame_verified"] = True
        record["input"].update(
            observation_file=str(observation_path), frame_index_file=str(run / "frame_index.json"),
            rgb_file=str(frame_path), rgb_sha256=hashlib.sha256(frame_path.read_bytes()).hexdigest(),
            simulation_seconds=capture_time,
            camera_eye=frame.get("camera_eye"),
            camera_world_yaw_degrees=frame.get("camera_world_yaw_degrees"),
            body_yaw_degrees=frame.get("body_yaw_degrees"),
            camera_pan_degrees=frame.get("camera_pan_degrees"),
            horizontal_fov_degrees=frame.get("horizontal_fov_degrees"),
        )
        sighting = {"time": capture_time, "feet": list(args.feet)}
        record["sighting"] = sighting

        # Imports are delayed so --help and missing-capture diagnostics work
        # without Isaac Sim or the production Skill's Python dependencies.
        from adaptive_motion import load_config
        from semantic_skills import SemanticSkills

        config = load_config(PROJECT / "configs" / "trial_rotating.json")
        isolated = SemanticSkills(config)
        request = {"observation_id": packet["observation_id"], "skill": "OBSERVE",
                   "params": {"sighting": sighting}}
        memory = isolated.validate(request, packet)
        record["validation_passed"] = True
        xy = memory.track["xy"]
        if (memory.track.get("frame") != args.frame_file
                or memory.track.get("seen_at") != capture_time
                or not isinstance(xy, list) or len(xy) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in xy)):
            raise ValueError("Validator did not project the selected frame to finite world xy")
        record["projection_passed"] = True
        record["world_xy_m"] = xy
        record["projection_assessment"] = (
            "Finite ground-plane xy within the formal validator's arena bounds; "
            "no independent position truth was supplied")
        record["status"] = "passed"
    except FileNotFoundError as exc:
        record["status"] = "blocked_missing_formal_capture"
        record["error"] = f"Missing formal run file: {exc.filename}"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Formal vision_trial run directory containing protocol.json")
    parser.add_argument("--observation-id", type=int, required=True)
    parser.add_argument("--frame-file", required=True,
                        help="Exact relative frames/...png path from the selected observation")
    parser.add_argument("--feet", type=float, nargs=2, required=True, metavar=("U", "V"),
                        help="Manually checked human ground-contact foot pixel")
    parser.add_argument("--annotation-note", required=True,
                        help="Brief explanation of why the RGB foot is visible and on the ground")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    destination = output_path(args.output_dir)
    result = check(args)
    destination.write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n",
                           encoding="utf-8")
    print(f"G1B1 {result['status']} | {destination}")
    if result["error"]:
        print(result["error"])
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
