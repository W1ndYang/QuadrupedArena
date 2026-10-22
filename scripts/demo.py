"""Isaac-rendered, scripted match presentation. No training or dynamics claims."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from isaaclab.app import AppLauncher

project = Path(__file__).resolve().parents[1]
from arena_paths import workspace_root
workspace = workspace_root()
parser = argparse.ArgumentParser()
parser.add_argument("--demo", action="store_true")
parser.add_argument("--tactical", action="store_true", help="Go2-W reactive cover exhibition")
parser.add_argument("--preview", action="store_true", help="Render five checkpoints instead of the full movie")
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--demo-config", type=Path)
parser.add_argument("--scene", type=Path, required=True, help="Generated rough-terrain USD; only required for legacy --demo mode")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.demo_config is None:
    args.demo_config=project/("configs/demo_tactical_v2.json" if args.tactical else "configs/demo_v1.json")
args.enable_cameras = True
app = AppLauncher(args).app

import carb
import omni.usd
import omni.timeline
import omni.replicator.core as rep
from pxr import UsdGeom, UsdShade, Gf, Sdf, Vt
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import imageio_ffmpeg
from demo_timeline import build
if args.tactical:
    from tactical_logic import build, block
from demo_actors import Actor
from beauty import prepare
from terrain import hill_mesh, gravel_layout


def curve(stage, path, color, width=.025):
    item = UsdGeom.BasisCurves.Define(stage, path)
    item.CreateTypeAttr("linear")
    item.CreateCurveVertexCountsAttr([2])
    item.CreatePointsAttr([(0., 0., 0.), (.001, 0., 0.)])
    item.CreateWidthsAttr([width])
    item.SetWidthsInterpolation("constant")
    item.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return item


def ground_sampler(config):
    spec = config["undulating"]
    mesh = hill_mesh(spec)
    nx = round(spec["size"][0]/spec["grid_spacing"])+1
    ny = round(spec["size"][1]/spec["grid_spacing"])+1
    heights = mesh.vertices[:, 2].reshape(nx, ny)
    stones = gravel_layout(config["gravel"], config["seed"])

    def height(xy):
        answer = 0.
        for center in spec["centers"]:
            local = np.asarray(xy)-center+np.asarray(spec["size"])/2
            if np.all(local >= 0) and np.all(local <= spec["size"]):
                ij = local/spec["grid_spacing"]
                i, j = np.minimum(np.floor(ij).astype(int), [nx-2, ny-2])
                a, b = ij-[i, j]
                answer = max(answer, float((1-a)*(1-b)*heights[i,j]+a*(1-b)*heights[i+1,j]+(1-a)*b*heights[i,j+1]+a*b*heights[i+1,j+1]))
        cfg = config["gravel"]
        for index, center in enumerate(cfg["centers"]):
            local = (np.asarray(xy)-center)*(1 if index == 0 else -1)
            if np.all(np.abs(local) < np.asarray(cfg["size"])/2):
                answer = max(answer, cfg["bed_height"])
                for x, y, radius, h in stones:
                    if np.linalg.norm(local-[x,y]) < radius:
                        answer = max(answer, cfg["bed_height"]+h)
        return answer
    return height


def chapter(state):
    t = state["t"]
    if t < 4:
        return "01  开局就位", "双方各 3 点生命；P1 激活。蓝色：四足 A，橙色：人形 B。"
    if t < 28:
        return "02  绕行与地形", "沿预设路线通过起伏路段、绕过掩体，接近任务点。"
    if t < 48:
        return "03  占点与虚拟标记", "独占激活区可计分；标记需持续锁定且视线不穿过掩体。"
    if t < 60:
        return "04  离点转移", "离开或双方争夺时停止计分；为下一任务点调整位置。"
    if t < 70:
        return "05  P2 激活", "比赛第 60 秒切换到 P2；P1 立即停止计分。"
    if t < 92:
        return "06  争夺与再机动", "虚拟标记后暂停标记和占点 3 秒，仍可继续移动。"
    if t < 120:
        return "07  最后转移", "利用通道重新接近激活点；最终按积分、剩余生命结算。"
    return "08  到时结算", "本段为编排示例，不作为机器人性能或赛制有效性的结论。"


def panel(rgb, state, events, intro=False, outro=False):
    width, height = rgb.size
    image = Image.new("RGB", (width, height+224), "#101d2d")
    image.paste(rgb, (0, 96))
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    fonts = {s: ImageFont.truetype(font_path, s) for s in (16,18,20,24,32)}
    cyan, orange, white = "#49c9ff", "#ffb360", "#e9f2fb"
    draw.text((22, 9), "四足 × 人形｜掩体对抗流程 Demo", font=fonts[24], fill=white)
    draw.text((width-265, 14), "Isaac Sim 原生场景渲染", font=fonts[18], fill="#aec3d8")
    for i, (name, color, x) in enumerate((("四足 A",cyan,22),("人形 B",orange,595))):
        draw.text((x, 53), f"{name}   HP {state['hp'][i]}/3    积分 {state['score'][i]:05.1f}", font=fonts[20], fill=color)
    draw.text((365, 55), f"{state['t']:05.1f} / 120 s", font=fonts[20], fill=white)
    y = height+108
    title, explanation = chapter(state)
    if intro:
        title, explanation = "比赛流程演示｜120 秒比赛 · 4 倍速播放", "预设路径 + 关节动画，无训练；人形对手也由脚本驱动。"
    if outro:
        winner = events[-1]["winner"]
        title = f"比赛结束｜{'四足 A 胜出' if winner == 'A' else '人形 B 胜出' if winner == 'B' else '平局'}（本次编排）"
        explanation = f"占点积分 A {state['score'][0]:.1f} : B {state['score'][1]:.1f}；剩余生命 {state['hp'][0]} : {state['hp'][1]}。"
    draw.text((22,y), title, font=fonts[24], fill=white)
    draw.text((22,y+36), explanation, font=fonts[18], fill="#b8cee0")
    draw.text((22,y+68), "脚本运动 / 非自主策略 / 非动力学验证 / 赛制草案；字幕与HUD为后期叠加", font=fonts[16], fill="#8da5bd")
    draw.rectangle((22,y+103,width-22,y+108), fill="#2c4155")
    draw.rectangle((22,y+103,22+(width-44)*state["t"]/120,y+108), fill=cyan)
    recent = [e for e in events if e["type"] == "mark" and 0 <= state["t"]-e["time"] < 4]
    active_text = f"P{state['point']+1} 激活"
    if state["contested"]:
        active_text += " · 争夺中，停止计分"
    elif any(state["scoring"]):
        active_text += " · " + ("四足 A" if state["scoring"][0] else "人形 B") + "占点 +1分/秒"
    if state["pause"] > 0:
        active_text = f"命中暂停：{state['pause']:.1f}s（仍可移动）"
    draw.rounded_rectangle((22,112,445,148), radius=8, fill="#132539")
    draw.text((34,117), active_text, font=fonts[18], fill=white)
    if recent:
        event = recent[-1]
        draw.rounded_rectangle((width-300,112,width-22,148), radius=8, fill="#633e25")
        draw.text((width-286,117), f"虚拟标记 {event['attacker']} → {event['target']}  生命 -1", font=fonts[18], fill="#fff0d6")
    if intro or outro:
        draw.rounded_rectangle((165,335,width-165,470), radius=18, fill="#14293e")
        draw.text((195,355), "预设路线 · 比赛流程演示" if intro else "时间到｜按积分结算", font=fonts[32], fill=white)
        draw.text((195,412), "绕行 → 占点 → 标记 → 切换 → 结算" if intro else explanation, font=fonts[18], fill="#a9d9ee")
    return image


def main():
    started = time.monotonic()
    config = json.loads(args.demo_config.read_text())
    arena = json.loads((project/"configs/arena_v0.json").read_text())
    terrain = json.loads((project/"configs/terrain_v1.json").read_text())
    states, events, validation = build(config, arena)
    for name, data in (("demo_config.json",config),("events.json",events),("timeline.json",states)):
        (args.run_dir/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n")
    omni.usd.get_context().open_stage(str(args.scene))
    for _ in range(15):
        app.update()
    stage = omni.usd.get_context().get_stage()
    # This presentation has no physical actors and never advances physics time.
    for path in ("/World/Probes", "/World/Robot", "/physicsScene"):
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(path)
    scene_dir = workspace/"simulators/scenes/quadruped-arena"/args.run_dir.name
    scene_dir.mkdir(parents=True, exist_ok=True)
    prepare(stage, scene_dir/"textures")
    settings = carb.settings.get_settings()
    settings.set("/rtx/pathtracing/spp", 8)
    settings.set("/rtx/pathtracing/totalSpp", 32)
    settings.set("/rtx/pathtracing/clampSpp", 32)
    actors = [Actor(stage, "/World/DemoActors/A", "go2w" if args.tactical else "go2", workspace), Actor(stage, "/World/DemoActors/B", "g1", workspace)]
    print(f"ACTORS loaded: {[a.mesh_count for a in actors]}", flush=True)
    colors = [(0.04,.65,1.), (1.,.35,.06)]
    trails = [curve(stage, f"/World/DemoOverlay/Trail{i}", c, .025) for i,c in enumerate(colors)]
    rings = [curve(stage, f"/World/DemoOverlay/Ring{i}", c, .035) for i,c in enumerate(colors)]
    point_rings = [curve(stage, f"/World/DemoOverlay/Point{i}", (.05,.9,.35), .045) for i in range(2)]
    beam = curve(stage, "/World/DemoOverlay/VirtualMark", (1.,.7,.12), .025)
    point_shaders = []
    for i in range(2):
        point_shaders.append(UsdShade.Shader(stage.GetPrimAtPath(f"/World/Tasks/P{i+1}/geometry/material/Shader")))
    if args.tactical:
        from tactical_visuals import panel as draw_panel, shot
        camera_prim=UsdGeom.Camera.Define(stage,"/World/TacticalCamera")
        camera_prim.CreateFocalLengthAttr(25.)
        camera_op=camera_prim.AddTransformOp()
        camera=str(camera_prim.GetPath())
    else:
        draw_panel=panel
        camera = rep.create.camera(position=(12.,-14.,15.5), look_at=(0.,0.,.1), focal_length=29.)
    product = rep.create.render_product(camera, tuple(config["resolution"]))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([product])
    rep.orchestrator.set_capture_on_play(False)
    ground = ground_sampler(terrain)
    phase = [0.,0.]
    last_t = 0.
    if args.tactical:
        frame_times=([0.,3.5,7.8,10.,20.,24.,31.,36.,40.,states[-1]["t"]] if args.preview
                     else np.linspace(0,states[-1]["t"],round(states[-1]["t"]*config["fps"])+1).tolist())
    else:
        frame_times = [0., 35., 61., 74., 120.] if args.preview else np.linspace(0,120,301).tolist()
    writer = imageio.get_writer(str(args.run_dir/"demo.mp4"), fps=config["fps"], codec="libx264", quality=8, macro_block_size=1, ffmpeg_log_level="error")
    count = 0
    poster = None
    try:
        for index,t in enumerate(frame_times):
            state = states[min(round(t/.02), len(states)-1)]
            if args.tactical:
                eye,target,_=shot(state)
                camera_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse())
            for i,actor in enumerate(actors):
                xy = state["xy"][i]
                phase[i] += (t-last_t)*state["speed"][i]*2*math.pi/.65
                actor.update(xy, state["yaw"][i], phase[i], state["speed"][i], ground(xy),
                             state["wheels"] if args.tactical and i==0 else None)
                angles = np.linspace(0,2*math.pi,49)
                pts = [(xy[0]+.40*math.cos(a),xy[1]+.40*math.sin(a),ground(xy)+.03) for a in angles]
                rings[i].GetPointsAttr().Set(pts)
                rings[i].GetCurveVertexCountsAttr().Set([len(pts)])
                history = states[:min(round(t/.02)+1,len(states)):25]
                if args.tactical: history=history[-12:]
                pts = [(s["xy"][i][0],s["xy"][i][1],ground(s["xy"][i])+.025) for s in history]
                if len(pts) < 2:
                    pts *= 2
                trails[i].GetPointsAttr().Set(pts)
                trails[i].GetCurveVertexCountsAttr().Set([len(pts)])
            for i,item in enumerate(point_rings):
                point = arena["task_points"][i]
                xy = point["center"]
                pts = [(xy[0]+point["radius"]*math.cos(a),xy[1]+point["radius"]*math.sin(a),.025) for a in np.linspace(0,2*math.pi,65)]
                item.GetPointsAttr().Set(pts)
                item.GetCurveVertexCountsAttr().Set([len(pts)])
                item.GetDisplayColorAttr().Set([Gf.Vec3f(*((.05,.8,.25) if i == state["point"] else (.3,.35,.4)))])
                if point_shaders[i]:
                    point_shaders[i].GetInput("diffuseColor").Set(Gf.Vec3f(*((.1,.55,.25) if i == state["point"] else (.20,.23,.26))))
            recent = any(e["type"] == "mark" and 0 <= t-e["time"] < 1.8 for e in events)
            if args.tactical and not state["los"]:
                hit=block((*state["xy"][1],1.05),(*state["xy"][0],.40),arena["covers"])
                beam.MakeVisible()
                beam.GetDisplayColorAttr().Set([Gf.Vec3f(1.,.12,.08)])
                beam.GetPointsAttr().Set([(*state["xy"][1],1.05),hit[2]])
            elif (recent or max(state["lock"]) > 0) and (not args.tactical or state["los"]):
                beam.MakeVisible()
                beam.GetDisplayColorAttr().Set([Gf.Vec3f(1.,.7,.12)])
                beam.GetPointsAttr().Set([(*state["xy"][0],ground(state["xy"][0])+.36),(*state["xy"][1],ground(state["xy"][1])+1.05)])
            else:
                beam.MakeInvisible()
            # Every USD transform update is rendered with paused simulation time.
            # Avoid Replicator's orchestrator barrier on a paused, physics-free stage.
            # Kit's renderer updates the attached RGB annotator directly.
            for _ in range(12 if index == 0 else 6):
                app.update()
            pixels = rgb.get_data()
            if pixels.size == 0 or np.std(pixels[...,:3]) < 1:
                raise RuntimeError(f"Blank frame at t={t}")
            raw = Image.fromarray(pixels[...,:3]).convert("RGB")
            framed = draw_panel(raw, state, events)
            if index == 0:
                intro = draw_panel(raw,state,events,intro=True)
                intro.save(args.run_dir/"intro.png")
                for _ in range(config["intro_seconds"]*config["fps"] if not args.preview else 1):
                    writer.append_data(np.asarray(intro)); count += 1
            writer.append_data(np.asarray(framed)); count += 1
            if index % 50 == 0 or args.preview:
                raw.save(args.run_dir/f"raw_{t:05.1f}.png")
                framed.save(args.run_dir/f"frame_{t:05.1f}.png")
                print(f"DEMO_FRAME {index+1}/{len(frame_times)} t={t:.1f} elapsed={time.monotonic()-started:.1f}s", flush=True)
            if index == len(frame_times)-1:
                poster = draw_panel(raw,state,events,outro=True)
                poster.save(args.run_dir/"poster.png")
                for _ in range(config["outro_seconds"]*config["fps"] if not args.preview else 1):
                    writer.append_data(np.asarray(poster)); count += 1
            last_t = t
    finally:
        writer.close()
        rgb.detach([product.path])
        product.destroy()
    stage.Export(str(scene_dir/"demo_final_pose.usda"))
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg,"-nostdin","-y","-v","error","-i",str(args.run_dir/"demo.mp4"),"-filter_complex",
                    "fps=10,scale=800:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=192[p];[s1][p]paletteuse=dither=sierra2_4a",
                    "-loop","0",str(args.run_dir/"demo.gif")], check=True, timeout=180)
    reader = imageio.get_reader(str(args.run_dir/"demo.mp4"))
    actual = reader.count_frames()
    reader.close()
    assert actual == count, (actual,count)
    with Image.open(args.run_dir/"demo.gif") as gif:
        duration_ms = 0
        for i in range(gif.n_frames):
            gif.seek(i); duration_ms += gif.info.get("duration",0)
    assert abs(duration_ms/1000-count/config["fps"]) < .3
    result = {"status":"passed","kind":"scripted_presentation_preview" if args.preview else "scripted_presentation",
              "renderer": settings.get("/rtx/rendermode"),"target_spp":32,"training":False,"dynamics_validated":False,
              "frames":count,"fps":config["fps"],"duration_seconds":count/config["fps"],"wall_seconds":time.monotonic()-started,
              "route_checks":validation,"events":events,"source_scene":str(args.scene),
              "outputs":{name:str(args.run_dir/name) for name in ("demo.mp4","demo.gif","poster.png","events.json","timeline.json")},
              "limitations":["Scripted root trajectories and URDF forward-kinematic gait, not a walking controller", "Both sides scripted; no actual teleoperation", "Mark requests choreographed; cover LOS and dwell/pause scoring computed", "Referee visibility uses fixed representative heights and cover boxes only", "Terrain following is visual approximation, not contact validation", "HUD added after Isaac frame capture"]}
    if args.tactical:
        result.update(variant="tactical_go2w_v2",rules_max_seconds=config["duration"],match_seconds=states[-1]["t"],
            limitations=["A uses rule-based partial-observation state transitions and preset routes, not a trained or general planner",
                "B is an intentionally scripted exhibition opponent; this is not a competitive benchmark",
                "Go2-W pose and wheel angles are kinematic animation, not a balance or contact controller",
                "Referee tests representative heights against cover boxes, not full mesh/terrain occlusion",
                "60-second exhibition cap and P2 at 30 seconds, not the earlier 120-second draft",
                "HUD, referee rays and trails are presentation overlays"])
    (args.run_dir/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("DEMO_RESULT "+json.dumps(result,ensure_ascii=False),flush=True)


try:
    main()
except BaseException:
    error = traceback.format_exc()
    (args.run_dir/"error.txt").write_text(error)
    (args.run_dir/"result.json").write_text(json.dumps({"status":"failed","error":error})+"\n")
    print(error,flush=True)
    raise
finally:
    app.close(wait_for_replicator=False)
