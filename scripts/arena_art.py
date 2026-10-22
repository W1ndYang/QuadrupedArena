"""Reusable Isaac/USD industrial courtyard art. Gameplay volumes stay separate.

No downloaded models, AI pictures or geometry displacement. CC0 surface maps
are stored in workspace/simulators/assets, not mixed into the source tree.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np


def beveled_box(size, bevel):
    """Convex render mesh whose AABB equals the original collision box."""
    from scipy.spatial import ConvexHull
    half = np.asarray(size, dtype=float) / 2
    bevel = min(bevel, float(min(half)) * .45)
    points = []
    for signs in itertools.product((-1, 1), repeat=3):
        for axis in range(3):
            p = half - bevel
            p[axis] = half[axis]
            points.append(p * signs)
    points = np.asarray(points)
    hull = ConvexHull(points)
    faces = hull.simplices.copy()
    for i, face in enumerate(faces):
        a, b, c = points[face]
        if np.dot(np.cross(b-a, c-a), hull.equations[i, :3]) < 0:
            faces[i] = face[::-1]
    return points, faces


def asset_manifest(asset_dir):
    assets = []
    for name, resolution in [('Concrete023', '2K'), ('Concrete034', '2K'), ('Asphalt012', '1K')]:
        prefix = asset_dir / name / f'{name}_{resolution}-JPG'
        files = [Path(str(prefix) + '_' + suffix + '.jpg') for suffix in ('Color', 'NormalGL', 'Roughness')]
        for p in files:
            if not p.is_file():
                raise FileNotFoundError(f'Missing PBR input: {p}')
        assets.append(dict(id=name, resolution=resolution, license='CC0-1.0',
            source=f'https://ambientcg.com/view?id={name}',
            download=f'https://ambientcg.com/get?file={prefix.name}.zip',
            license_url='https://docs.ambientcg.com/license/',
            files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    return assets


class Courtyard:
    def __init__(self, stage, assets, generated):
        from pxr import Gf, Sdf, UsdGeom, UsdShade
        self.stage, self.assets, self.generated = stage, Path(assets), Path(generated)
        self.generated.mkdir(parents=True, exist_ok=True)
        self.Gf, self.Sdf, self.UsdGeom, self.UsdShade = Gf, Sdf, UsdGeom, UsdShade
        self.materials, self.serial = {}, 0
        self.manifest = asset_manifest(self.assets)
        self.material('concrete', (.63, .62, .58), texture='Concrete034', scale=1.3)
        self.material('pale', (.78, .77, .71), texture='Concrete034', scale=1.3)
        self.material('weathered', (.55, .53, .48), texture='Concrete023', scale=1.4)
        self.material('asphalt', (.40, .43, .46), texture='Asphalt012', scale=.6)
        self.material('dark', (.035, .045, .05), roughness=.6, metallic=.55)
        self.material('metal', (.24, .27, .28), roughness=.45, metallic=.8)
        self.material('paint', (.71, .42, .075), texture='Concrete023', scale=.8, roughness=.75)
        self.material('white', (.62, .65, .62), roughness=.8)
        self.material('blue', (.018, .045, .055), roughness=.65, metallic=.3)
        self.material('joint', (.20, .20, .18), roughness=.95)
        self.material('rust', (.20, .075, .035), roughness=.95)
        self.material('glass', (.025, .07, .085), roughness=.16, metallic=.7)
        self.material('lamp', (1., .85, .6), emission=25.)

    def material(self, name, color, texture=None, scale=1., roughness=.85, metallic=0., emission=0., image=None):
        import isaacsim
        Gf, Sdf, UsdShade = self.Gf, self.Sdf, self.UsdShade
        mat = UsdShade.Material.Define(self.stage, '/World/ArtMaterials/' + name)
        sh = UsdShade.Shader.Define(self.stage, str(mat.GetPath()) + '/Shader')
        sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        sh.SetSourceAsset(Sdf.AssetPath(str(Path(isaacsim.__file__).parent / 'kit/mdl/core/Base/OmniPBR.mdl')), 'mdl')
        sh.SetSourceAssetSubIdentifier('OmniPBR', 'mdl'); sh.CreateOutput('out', Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput('mdl').ConnectToSource(sh.ConnectableAPI(), 'out')
        values = dict(diffuse_color_constant=('Color3f', Gf.Vec3f(*color)),
            reflection_roughness_constant=('Float', roughness), metallic_constant=('Float', metallic))
        if texture:
            res = '1K' if texture == 'Asphalt012' else '2K'
            base = self.assets / texture / f'{texture}_{res}-JPG'
            values.update(diffuse_tint=('Color3f', Gf.Vec3f(*color)),
                diffuse_texture=('Asset', Sdf.AssetPath(str(base)+'_Color.jpg')),
                normalmap_texture=('Asset', Sdf.AssetPath(str(base)+'_NormalGL.jpg')),
                reflectionroughness_texture=('Asset', Sdf.AssetPath(str(base)+'_Roughness.jpg')),
                reflection_roughness_texture_influence=('Float', 1.),
                bump_factor=('Float', .22), flip_tangent_v=('Bool', False),
                project_uvw=('Bool', True), world_or_object=('Bool', False),
                texture_scale=('Float2', Gf.Vec2f(scale)))
        if image:
            values['diffuse_texture'] = ('Asset', Sdf.AssetPath(str(image)))
            values['diffuse_tint'] = ('Color3f', Gf.Vec3f(1))
        if emission:
            values.update(enable_emission=('Bool', True), emissive_color=('Color3f', Gf.Vec3f(*color)), emissive_intensity=('Float', emission))
        for key, (kind, value) in values.items(): sh.CreateInput(key, getattr(Sdf.ValueTypeNames, kind)).Set(value)
        self.materials[name] = mat
        return mat

    def bind(self, item, material):
        self.UsdShade.MaterialBindingAPI.Apply(item.GetPrim()).Bind(self.materials[material])

    def box(self, path, size, position, material, bevel=.018, collision=False):
        from pxr import UsdPhysics, Vt
        UsdGeom, Gf = self.UsdGeom, self.Gf
        root = UsdGeom.Xform.Define(self.stage, path)
        root.AddTranslateOp().Set(Gf.Vec3d(*position))
        points, faces = beveled_box(size, bevel)
        mesh = UsdGeom.Mesh.Define(self.stage, path+'/Visual')
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr([3]*len(faces)); mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr('none'); self.bind(mesh, material)
        if collision:
            collider = UsdGeom.Cube.Define(self.stage, path+'/Collider')
            collider.CreateSizeAttr(1.); collider.AddScaleOp().Set(Gf.Vec3f(*size))
            collider.CreateVisibilityAttr('invisible')
            UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
            physics_mat=self.UsdShade.Material.Define(self.stage,'/World/PhysicsMaterials/Arena')
            physics_api=UsdPhysics.MaterialAPI.Apply(physics_mat.GetPrim())
            physics_api.CreateStaticFrictionAttr(.8);physics_api.CreateDynamicFrictionAttr(.7)
            self.UsdShade.MaterialBindingAPI.Apply(collider.GetPrim()).Bind(physics_mat,materialPurpose='physics')
        return root

    def detail(self, size, position, material, bevel=.008, root='/World/Art'):
        self.serial += 1
        return self.box(f'{root}/Part{self.serial:04}', size, position, material, bevel)

    def lines(self, path, segments, material, width=.012):
        curve = self.UsdGeom.BasisCurves.Define(self.stage, path)
        curve.CreateTypeAttr('linear'); curve.CreateCurveVertexCountsAttr([len(s) for s in segments])
        curve.CreatePointsAttr([p for s in segments for p in s]); curve.CreateWidthsAttr([width])
        curve.SetWidthsInterpolation('constant'); self.bind(curve, material)
        return curve

    def painted_ring(self,path,center,radius,width=.035):
        x,y=center;points=[];indices=[]
        for angle in np.linspace(0,2*math.pi,97):
            for r in (radius-width/2,radius+width/2):
                points.append((x+r*math.cos(angle),y+r*math.sin(angle),.004))
        for i in range(96):indices.extend([2*i,2*i+1,2*i+3,2*i+2])
        mesh=self.UsdGeom.Mesh.Define(self.stage,path)
        mesh.CreatePointsAttr(points);mesh.CreateFaceVertexCountsAttr([4]*96)
        mesh.CreateFaceVertexIndicesAttr(indices);mesh.CreateSubdivisionSchemeAttr('none')
        self.bind(mesh,'paint')

    def sign(self, path, title, subtitle, center, width, height, plane='wall', color=(205, 170, 90)):
        """Authored typography as a texture asset, rendered by Isaac on a USD quad."""
        from PIL import Image, ImageDraw, ImageFont
        from pxr import Vt
        W, H = 1024, max(128, int(1024*height/width))
        im = Image.new('RGB', (W,H), (24,32,35)); d = ImageDraw.Draw(im)
        font = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
        d.rectangle((0,0,18,H), fill=color)
        fs = min(int(H*.47), int((W-80)/max(1,len(title)) * 1.4))
        d.text((44,int(H*.08)), title, fill=(221,220,204), font=ImageFont.truetype(font, fs))
        d.text((47,int(H*.70)), subtitle, fill=color, font=ImageFont.truetype(font, max(14,int(H*.14))))
        file = self.generated/(path.rsplit('/',1)[-1]+'.png'); im.save(file)
        name = 'Label_'+path.rsplit('/',1)[-1]; self.material(name, (1,1,1), image=file)
        x,y,z = center; w,h = width/2,height/2
        points = [(x-w,y,z-h),(x+w,y,z-h),(x+w,y,z+h),(x-w,y,z+h)] if plane=='wall' else [(x-w,y-h,z),(x+w,y-h,z),(x+w,y+h,z),(x-w,y+h,z)]
        if plane=='wall_x':points=[(x,y+w,z-h),(x,y-w,z-h),(x,y-w,z+h),(x,y+w,z+h)]
        mesh = self.UsdGeom.Mesh.Define(self.stage,path)
        mesh.CreatePointsAttr(points); mesh.CreateFaceVertexCountsAttr([4]);mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
        mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
        self.UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',self.Sdf.ValueTypeNames.TexCoord2fArray,'faceVarying').Set(Vt.Vec2fArray([(0,0),(1,0),(1,1),(0,1)]))
        self.bind(mesh,name)

    def build(self, cfg):
        # Gameplay: same boxes and target circles as the existing R0 arena.
        self.box('/World/Floor', cfg['arena']['floor_size'], (0,0,-.1), 'asphalt', .005, True)
        for i,c in enumerate(cfg['arena']['covers']):
            root = '/World/Covers/'+c['name']; sx,sy,sz = c['size']
            self.box(root,c['size'],(*c['center'],sz/2),'concrete' if sz>1 else 'pale',.025,True)
            # Flush bands and ties are visual-only children; they follow the collapse.
            long_x = sx>sy
            self.detail((sx+.002,sy+.002,.10),(0,0,-sz/2+.16),'paint',.002,root)
            self.detail((sx+.002,sy+.002,.055),(0,0,sz/2-.08),'dark',.002,root)
            for v in (-.32,.32):
                for h in ([.42,-.42] if sz>1 else [0.]):
                    size=(.045,.003,.045) if long_x else (.003,.045,.045)
                    pos=(v*sx,-sy/2-.002,h) if long_x else (-sx/2-.002,v*sy,h)
                    self.detail(size,pos,'dark',.008,root)
            if long_x:
                self.sign(root+'/ID'+c['name'],c['name'].upper(),'R0 / COVER', (0,-sy/2-.003,0), min(.75,sx*.65),min(.30,sz*.45))
            else:
                self.sign(root+'/ID'+c['name'],c['name'].upper(),'R0 / COVER', (-sx/2-.003,0,0),.75,.30,'wall_x')
            if sz>1:
                for sign in (-1,1):
                    size=(.008,.002,sz-.10) if long_x else (.002,.008,sz-.10)
                    pos=(sign*sx*.25,-sy/2-.001,0) if long_x else (-sx/2-.001,sign*sy*.25,0)
                    self.detail(size,pos,'joint',.0005,root)
        # Expand the visual compound to 18 x 14 m, not the legal match region.
        for size,pos in [((18,3,.18),(0,-5.5,-.095)),((18,3,.18),(0,5.5,-.095)),((3,8,.18),(-7.5,0,-.095)),((3,8,.18),(7.5,0,-.095))]:
            self.detail(size,pos,'concrete')
        self.detail((180,180,.2),(0,0,-.26),'asphalt')
        seams=[]
        for x in np.arange(-9,9.1,1.5):
            for ya,yb in [(-7,-4),(4,7)]:seams.append([(x,ya,.002),(x,yb,.002)])
        for y in np.arange(-7,7.1,1.5):
            for xa,xb in [(-9,-6),(6,9)]:seams.append([(xa,y,.002),(xb,y,.002)])
        self.lines('/World/Art/ExpansionJoints',seams,'dark',.013)
        # Painted boundary; deliberately not a new impassable rail.
        for y in (-4,4):self.detail((12,.07,.004),(0,y,.004),'paint',.0005)
        for x in (-6,6):self.detail((.07,8,.004),(x,0,.004),'paint',.0005)
        for x in np.arange(-5.7,5.8,.65):
            for y in (-4.18,4.18):self.detail((.28,.06,.003),(x,y,.003),'white',.0005)
        for i,p in enumerate(cfg['arena']['task_points']):
            x,y=p['center']; r=p['radius']
            self.painted_ring('/World/Art/Target'+str(i),(x,y),r)
            self.sign('/World/Art/TargetLabel'+str(i),p['name'],'HOLD / SCORE',(x,y,.012),.58,.26,'floor')
        self.sign('/World/Art/FloorEntry','R0 / LIVE ZONE','AUTHORIZED ROBOTICS TEST AREA',(0,-5.2,.012),4.8,.72,'floor')
        # Rear perimeter: buttressed concrete, utility pipes, steel doors.
        for x in np.arange(-7.5,7.6,3):
            self.detail((2.96,.32,2.8),(x,7,1.4),'concrete')
            self.detail((.22,.48,3.02),(x-1.5,7,1.51),'pale')
            self.detail((2.9,.40,.12),(x,7,2.86),'dark')
        for x in (-9,9):
            self.detail((.32,14,1.0),(x,0,.5),'weathered')
            self.detail((.40,14,.10),(x,0,1.05),'dark')
            wire=[]
            for y in np.arange(-7,7.01,2):self.detail((.065,.065,1.55),(x,y,1.8),'metal')
            for y in np.arange(-7,7,.16):
                end=min(7,y+1.3);wire.append([(x,y,1.12),(x,end,1.12+end-y)])
                end=min(7,y+1.3);wire.append([(x,y,2.42),(x,end,2.42-end+y)])
            self.lines(f'/World/Art/Fence{int(x+9)}',wire,'metal',.007)
            self.lines(f'/World/Art/FenceRail{int(x+9)}',[[(x,-7,2.45),(x,7,2.45)]],'metal',.045)
        for x in (-6,6):
            self.detail((5.9,.32,.85),(x,-7,.425),'concrete')
            self.detail((5.95,.4,.08),(x,-7,.89),'dark')
        for x in (-3,3):
            self.detail((.3,.3,3.3),(x,-7,1.65),'dark')
            self.detail((.37,.37,.08),(x,-7,3.31),'paint')
        # Industrial gate and observation/control building beyond the match.
        self.detail((3,.12,2.25),(5,6.78,1.125),'blue')
        for x in np.arange(3.6,6.45,.16):self.detail((.025,.035,2.1),(x,6.70,1.12),'metal',.002)
        self.sign('/World/Art/GateSign','02 / SERVICE','NO UNAUTHORIZED ACCESS',(5,6.69,2.55),2.7,.45)
        self.detail((8,4,4.4),(-3,9.2,2.2),'pale',.04)
        self.detail((8.25,4.2,.18),(-3,9.2,4.49),'dark')
        self.detail((7.1,.08,.85),(-3,7.17,3.35),'dark')
        for x in np.arange(-6.1,.2,1.04):
            self.detail((.89,.025,.65),(x,7.115,3.35),'glass')
        self.sign('/World/Art/MainSign','R0 / PROVING GROUND','AUTONOMOUS SYSTEMS     /     SECTOR 07',(-3,6.78,2.1),5.9,.83)
        for x in (-5.8,-4.2):
            self.detail((1.1,1.1,.75),(x,9.2,4.95),'metal')
            for z in np.arange(4.7,5.2,.09):self.detail((1.02,.035,.025),(x,8.63,z),'dark',.001)
        # Side cabinets, conduits, and drain grates are outside the playable box.
        for x,y in [(7.7,4.7),(7.7,3.4),(-7.8,5.5)]:
            self.detail((.65,.8,1.35),(x,y,.675),'blue')
            self.detail((.04,.3,.03),(x-.34,y,.9),'metal')
            for z in np.arange(.3,.7,.07):self.detail((.015,.52,.022),(x-.335,y,z),'dark',.001)
        for x in (-6.5,6.5):
            self.detail((.28,9,.02),(x,0,.008),'dark')
            for y in np.arange(-4.45,4.5,.09):self.detail((.25,.02,.021),(x,y,.018),'metal',.001)
        self.lines('/World/Art/Pipe',[[(-8.78,-5,.45),(-8.78,5.8,.45),(-8.78,5.8,2.4),(-6,5.8,2.4)]],'rust',.06)
        for x in (-8.3,8.3):
            for y in (-5.5,5.5):
                self.detail((.12,.12,4.5),(x,y,2.25),'metal')
                self.detail((.8,.22,.10),(x,y,4.5),'dark')
                self.detail((.65,.18,.018),(x,y,4.44),'lamp')
        # Background massing prevents the low camera looking onto an empty plane.
        for x,y,sx,sy,h in [(18,12,12,14,6.5),(-15,17,9,12,5.2),(4,24,19,10,7.0)]:
            self.detail((sx,sy,h),(x,y,h/2),'concrete',.05)
            self.detail((sx+.25,sy+.25,.22),(x,y,h+.11),'dark')
            for px in np.arange(x-sx/2+.8,x+sx/2,1.4):
                self.detail((.10,.07,h-.2),(px,y-sy/2-.04,h/2),'metal')
            self.detail((sx*.65,.04,.7),(x,y-sy/2-.08,h-1.25),'glass')
        return {'playable_size_m':[12,8], 'compound_size_m':[18,14],
            'gameplay_changed':False, 'decorative_collision':False,
            'assets':self.manifest, 'decorative_parts':self.serial,
            'displacement_enabled':False, 'normal_maps':'OpenGL, flip_tangent_v=False'}


def lighting(stage, samples=256):
    import carb
    from pxr import UsdLux, UsdGeom, Gf
    sky=UsdLux.DomeLight.Define(stage,'/World/Lighting/Sky')
    sky.CreateIntensityAttr(900.);sky.CreateColorAttr(Gf.Vec3f(.78,.87,1.))
    sun=UsdLux.DistantLight.Define(stage,'/World/Lighting/Sun')
    sun.CreateIntensityAttr(2200.);sun.CreateAngleAttr(1.4);sun.CreateColorAttr(Gf.Vec3f(1.,.91,.80))
    UsdGeom.Xformable(sun).AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(-8,-12,18),Gf.Vec3d(0,0,0),Gf.Vec3d(0,0,1)).GetInverse())
    settings=carb.settings.get_settings()
    for key,value in {'/rtx/rendermode':'PathTracing','/rtx/pathtracing/spp':8,
        '/rtx/pathtracing/totalSpp':samples,'/rtx/pathtracing/clampSpp':samples,
        '/rtx/pathtracing/maxBounces':6,'/rtx/pathtracing/optixDenoiser/enabled':True,
        '/rtx/post/histogram/enabled':False}.items():settings.set(key,value)
