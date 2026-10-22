"""Original container yard: native USD structures and locally licensed PBR art.

Geometry is not a textured backdrop: container panels, ramps and the bridge
deck are individually collidable. Small ribs, labels and utilities are visual.
"""
import hashlib
import math
from pathlib import Path
import numpy as np
from arena_art import Courtyard
from dockyard_geometry import solids, ramp_height


class Dockyard(Courtyard):
    def __init__(self, stage, assets, generated):
        super().__init__(stage, assets, generated)
        self.material('steel', (.39,.43,.45), 'Metal032', scale=1.4, metallic=.65)
        self.material('cargo_teal', (.09,.37,.39), 'Metal032', scale=1.2, metallic=.25)
        self.material('cargo_red', (.51,.15,.09), 'Metal032', scale=1.2, metallic=.22)
        self.material('cargo_ochre', (.68,.43,.13), 'Metal032', scale=1.2, metallic=.22)
        self.material('slab', (.72,.72,.66), 'Concrete034', scale=.8)
        self.material('tube', (.72,.86,1.), emission=5000.)
        self.material('cabinet', (.62,.67,.64), 'Metal032', scale=1.2, metallic=.18)
        self.material('crate', (.49,.32,.14), 'Metal032', scale=1.5, metallic=.12)
        self.material('rubber', (.026,.032,.034), roughness=.94)
        files=[self.assets/'Metal032'/('Metal032_2K-JPG_'+k+'.jpg') for k in ('Color','NormalGL','Roughness')]
        self.manifest.append(dict(id='Metal032',resolution='2K',license='CC0-1.0',
            source='https://ambientcg.com/view?id=Metal032',license_url='https://docs.ambientcg.com/license/',
            files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))

    def ramp(self, s):
        from pxr import UsdPhysics, Vt
        from scipy.spatial import ConvexHull
        points=np.array(s['points'],float); hull=ConvexHull(points); faces=hull.simplices.copy()
        for i,face in enumerate(faces):
            a,b,c=points[face]
            if np.dot(np.cross(b-a,c-a),hull.equations[i,:3])<0:faces[i]=face[::-1]
        mesh=self.UsdGeom.Mesh.Define(self.stage,'/World/Geometry/'+s['name'])
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr('none');self.bind(mesh,s['material'])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
        return mesh

    def beam(self, p1, p2, width, material, root='/World/Art', collision=False):
        a,b=np.array(p1,float),np.array(p2,float); d=b-a
        self.serial+=1
        result=self.box(f'{root}/Beam{self.serial:04}',(width,width,float(np.linalg.norm(d))),
            (a+b)/2,material,.006,collision)
        rotation=self.Gf.Rotation(self.Gf.Vec3d(0,0,1),self.Gf.Vec3d(*d))
        result.AddOrientOp(precision=self.UsdGeom.XformOp.PrecisionDouble).Set(rotation.GetQuat())
        return result

    def walls(self, d):
        for w in d['walls']:
            if w.get('kind'):
                self.facility(w)
                continue
            sx,sy,h=w['size'];root='/World/Geometry/'+w['name'];longx=sx>sy
            self.detail((sx+.003,sy+.003,.09),(0,0,-h/2+.15),'paint',.001,root)
            self.detail((sx+.003,sy+.003,.055),(0,0,h/2-.055),'dark',.001,root)
            for side in [-1,1]:
                for t in [-.3,.3]:
                    pos=(t*sx,side*(sy/2+.003),0) if longx else (side*(sx/2+.003),t*sy,0)
                    size=(.038,.004,.038) if longx else (.004,.038,.038)
                    self.detail(size,pos,'dark',.003,root)
                    if h>1:
                        pos=(t*sx,side*(sy/2+.002),0) if longx else (side*(sx/2+.002),t*sy,0)
                        size=(.008,.003,h-.12) if longx else (.003,.008,h-.12)
                        self.detail(size,pos,'joint',.001,root)
            pos=(0,-sy/2-.004,0) if longx else (-sx/2-.004,0,0)
            title='C-20 / FALL ZONE' if w['name']==d['collapse']['wall'] else w['name'].upper()
            self.sign(root+'/ID'+w['name'],title,'DOCK 07 / MODULAR COVER',pos,
                min(1.2,max(sx,sy)*.72),min(.34,h*.44),'wall' if longx else 'wall_x')

    def facility(self,w):
        """Port equipment built inside the original cover-height envelope.

Main closed equipment uses one structural box; hinges/vents/straps are visual.
The pump-house facade shares the collapse transform, its other panels do not.
"""
        sx,sy,h=w['size'];root='/World/Geometry/'+w['name'];kind=w['kind']
        longx=sx>=sy;L,D=(sx,sy) if longx else (sy,sx)
        def point(a,b,z):return (a,b,z-h/2) if longx else (b,a,z-h/2)
        def part(size,pos,material,bevel=.004):
            la,de,hi=size;return self.detail((la,de,hi) if longx else (de,la,hi),point(*pos),material,bevel,root)
        def label(title,sub,z,width=1.2,height=.23):
            self.sign(root+'/Label'+w['name'],title,sub,point(0,-D/2-.052,z),min(width,L-.12),height,'wall' if longx else 'wall_x')
        part((L,D,.12),(0,0,.06),'dark')
        if kind in ('power_bank','cooler'):
            n=3 if kind=='power_bank' else 2
            for i in range(n):
                a=-L/2+(i+.5)*L/n
                part((L/n-.04,.022,h-.22),(a,-D/2-.01,h/2+.02),'cabinet')
                part((.018,.032,h-.24),(a-L/n*.46,-D/2-.02,h/2),'dark')
                if kind=='power_bank':
                    part((.035,.055,.24),(a+L/n*.28,-D/2-.035,.9),'dark')
                    part((L/n*.48,.03,.14),(a,-D/2-.03,1.55),'blue')
                    for z in np.arange(.28,.65,.07):part((L/n*.62,.04,.022),(a,-D/2-.028,z),'dark')
                else:
                    # Circular fan housings, grilles and blades are native geometry.
                    self.serial+=1
                    fan=self.UsdGeom.Cylinder.Define(self.stage,root+f'/Fan{self.serial}')
                    fan.CreateRadiusAttr(min(.32,L/n*.40));fan.CreateHeightAttr(.03)
                    fan.CreateAxisAttr('Y' if longx else 'X')
                    fan.AddTranslateOp().Set(self.Gf.Vec3d(*point(a,-D/2-.035,1.31)));self.bind(fan,'dark')
                    segments=[]
                    radius=min(.29,L/n*.37)
                    for angle in np.linspace(0,2*math.pi,17):
                        segments.append([point(a+radius*math.cos(angle),-D/2-.057,1.31+radius*math.sin(angle)),point(a,-D/2-.057,1.31)])
                    self.lines(root+f'/FanGrille{self.serial}',segments,'metal',.012)
                    for z in np.arange(.28,.78,.07):part((L/n*.75,.04,.026),(a,-D/2-.028,z),'dark')
            part((L,D,.07),(0,0,h-.035),'steel')
            label('SHORE POWER' if kind=='power_bank' else 'REFRIGERATION','PORT SERVICES / '+w['name'],h-.26,1.65,.21)
        elif kind=='service_hut':
            # Compact machinery enclosure, not a human-height office building.
            for a in np.arange(-L/2+.1,L/2,.20):
                for side in [-1,1]:part((.045,.03,h-.18),(a,side*(D/2+.01),h/2),'metal')
            part((1.05,.045,1.72),(-.37,-D/2-.025,.94),'dark')
            for z in np.arange(.18,1.68,.12):part((1.,.05,.035),(-.37,-D/2-.05,z),'steel')
            part((.48,.05,.30),(.75,-D/2-.035,1.43),'glass')
            part((L,D,.12),(0,0,h-.06),'dark')
            label('SERVICE MODULE','DOCK MAINTENANCE / 07',1.98,1.9,.20)
        elif kind=='pump_facade':
            part((.85,.035,1.65),(-.53,-D/2-.02,.91),'cargo_teal')
            part((.04,.05,.20),(-.23,-D/2-.045,.91),'metal')
            part((.67,.03,.56),(.62,-D/2-.022,1.15),'dark')
            for z in np.arange(.93,1.39,.07):part((.60,.04,.025),(.62,-D/2-.042,z),'steel')
            # Masonry courses visually connect this front wall to the small shed.
            segments=[]
            for z in np.arange(.22,h-.13,.25):
                segments.append([point(-L/2,-D/2-.003,z),point(L/2,-D/2-.003,z)])
            self.lines(root+'/MasonryJoints',segments,'joint',.009)
            label('PUMP STATION','C-20 / MAINTENANCE',1.97,1.65,.23)
        elif kind in ('freight_stack','pallet'):
            rows=2 if kind=='freight_stack' else 1
            cols=max(1,round(L/1.25));base=.14 if kind=='pallet' else .10
            for z in np.linspace(base,h,rows+1):part((L+.008,D+.008,.026),(0,0,min(h-.014,z)),'dark')
            for i in range(cols):
                a=-L/2+(i+.5)*L/cols
                for side in [-1,1]:
                    part((.055,.025,h-base),(a-L/cols*.44,side*(D/2+.008),(h+base)/2),'steel')
                    part((.055,.025,h-base),(a+L/cols*.44,side*(D/2+.008),(h+base)/2),'steel')
                    for z in np.linspace(base+.12,h-.12,3 if rows==1 else 7):
                        part((L/cols-.10,.015,.018),(a,side*(D/2+.005),z),'dark')
                # Straps wrap over the top but do not add to the original height.
                part((.042,D+.014,.016),(a,0,h-.008),'paint')
                for side in [-1,1]:part((.042,.02,h-base),(a,side*(D/2+.015),(h+base)/2),'paint')
            if kind=='pallet':
                for a in [-L*.3,L*.3]:part((L*.18,.025,.075),(a,-D/2-.015,.07),'rubber')
                label('PORT FREIGHT','PALLET / KEEP DRY',h*.59,1.1,.14)
            else:label('EXPORT / 07','SECURED FREIGHT / '+w['name'],h*.72,1.55,.26)
        elif kind=='tool_chest':
            part((L+.006,D+.006,.055),(0,0,h-.028),'steel')
            for a in [-L*.35,L*.35]:
                part((.07,.04,.12),(a,-D/2-.025,h-.13),'steel')
                part((.28,.05,.05),(a,-D/2-.03,h*.45),'dark')
            label('DOCK TOOLS','SERVICE / 07',h*.47,1.2,.16)
        elif kind=='fender':
            for a in np.linspace(-L*.40,L*.40,6):
                part((.10,D+.02,h-.10),(a,0,h/2),'dark',.015)
            for a in [-L*.34,L*.34]:part((.12,D+.025,.025),(a,0,h-.013),'paint')
            label('FENDER','BERTH 07',h*.5,.8,.14)

    def containers(self, d):
        from pxr import UsdLux
        for c in d['containers']:
            x,y=c['center']; L,W,H=c['length'],c['width'],c['height'];axis=c['axis'];mat=c['material']
            def local(size,pos,material=mat):
                a,b,z=pos;la,wi,hi=size
                return self.detail((la,wi,hi) if axis=='x' else (wi,la,hi),
                    (x+a,y+b,z) if axis=='x' else (x+b,y+a,z),material,.006)
            # Corrugation is deliberately geometric so low-angle views retain relief.
            for a in np.arange(-L/2+.22,L/2-.1,.30):
                for side in [-1,1]:
                    local((.09,.055,H-.25),(a,side*(W/2+.018),H/2))
                    local((.075,.035,H-.3),(a,side*(W/2-.16),H/2))
                local((.07,W-.1,.028),(a,0,H+.012))
            for side in [-1,1]:
                for end in [-1,1]:
                    local((.16,.17,H+.02),(end*(L/2-.08),side*(W/2-.05),H/2),'dark')
                for z in [.09,H-.06]:local((L,.12,.12),(0,side*(W/2+.015),z),'dark')
            for end in [-1,1]:
                local((.16,W,.13),(end*(L/2-.07),0,H-.10),'paint')
                if not c['open_ends']:
                    for across in [-.8,-.25,.25,.8]:
                        local((.05,.035,H-.25),(end*(L/2+.01),across,H/2),'steel')
                else:
                    local((.10,W-.3,.015),(end*(L/2-.07),0,.04),'paint')
            # Large external ID and visible internal port ID; no fake dark tunnel plane.
            center=(x,y-W/2-.05,1.55) if axis=='x' else (x-W/2-.05,y,1.55)
            self.sign('/World/Art/ContainerLabel'+c['name'],c['name']+' / '+('THROUGH' if c['open_ends'] else 'CARGO'),
                'DOCK 07    /    OPEN BOTH ENDS' if c['open_ends'] else 'SEALED CARGO / KEEP CLEAR',center,
                min(2.65,L*.65),.50,'wall' if axis=='x' else 'wall_x',color=(212,180,101))
            if c['open_ends']:
                for a in [-L*.24,L*.24]:
                    local((.65,.10,.035),(a,0,H-.18),'tube')
                    light=UsdLux.RectLight.Define(self.stage,'/World/Lighting/'+c['name']+'_'+str(int((a+L)*10)))
                    light.CreateIntensityAttr(5000.);light.CreateWidthAttr(1.2);light.CreateHeightAttr(.35)
                    light.CreateColorAttr(self.Gf.Vec3f(.82,.91,1.))
                    light.AddTranslateOp().Set(self.Gf.Vec3d(x+a if axis=='x' else x,y if axis=='x' else y+a,H-.26))

    def bridge(self, d):
        b=d['bridge'];h=b['height'];half=b['deck_length']/2;w=b['width'];r=b['ramp_run'];rail=b['rail_height']
        # Rails are collidable; the deck and sloped surfaces remain clear between them.
        for s in [-1,1]:
            y=s*(w/2+.055)
            for x in np.linspace(-half,half,6):
                self.beam((x,y,h),(x,y,h+rail),.055,'paint',collision=True)
            for height in [.45,rail]:self.beam((-half,y,h+height),(half,y,h+height),.055,'paint',collision=True)
            self.detail((b['deck_length'],.08,.11),(0,y,h+.035),'dark')
        for ramp in [s for s in solids(d) if s['shape']=='ramp']:
            low,high=ramp['low_x'],ramp['high_x']
            for side in [-1,1]:
                y=side*(w/2+.055)
                for x in np.linspace(low,high,7):
                    z=ramp_height(ramp,x);self.beam((x,y,z),(x,y,z+rail),.055,'paint',collision=True)
                for height in [.45,rail]:
                    self.beam((low,y,.025+height),(high,y,h+height),.055,'paint',collision=True)
            # Thin real anti-slip cross strips follow the slope, not a stair obstacle.
            for x in np.arange(min(low,high)+.12,max(low,high),.22):
                self.detail((.04,w-.14,.012),(x,0,ramp_height(ramp,x)+.009),'dark',.002)
        for x in np.arange(-half+.12,half,.22):self.detail((.04,w-.14,.009),(x,0,h+.006),'dark',.002)
        self.sign('/World/Art/BridgeID','07 / CROSSING','ELEVATED ROUTE / UNDERPASS CLEAR',
            (0,-w/2-.051,h-.11),2.2,.23)

    def ground(self,d):
        sx,sy=d['playable_size_m'];hx,hy=sx/2,sy/2
        self.box('/World/Floor',(sx,sy,.2),(0,0,-.1),'asphalt',.001,True)
        self.detail((100,100,.2),(0,0,-.22),'asphalt')
        # Surface patches are flush decals/slabs, not unmodelled movement obstacles.
        self.detail((19,5.4,.004),(0,0,.001),'slab',.0002)
        for y in [-9.7,9.7]:self.detail((30,3.8,.004),(0,y,.001),'weathered',.0002)
        for x in [-14.5,14.5]:self.detail((2.8,21,.004),(x,0,.001),'concrete',.0002)
        seams=[]
        for x in np.arange(-15,15.1,2.5):
            for a,b in [(-11.6,-7.8),(-2.7,2.7),(7.8,11.6)]:seams.append([(x,a,.005),(x,b,.005)])
        self.lines('/World/Art/SlabJoints',seams,'joint',.012)
        for y in [-hy+.16,hy-.16]:self.detail((sx-.32,.075,.004),(0,y,.006),'paint',.0005)
        for x in [-hx+.16,hx-.16]:self.detail((.075,sy-.32,.004),(x,0,.006),'paint',.0005)
        for y in [-7.15,7.15]:
            for x in np.arange(-13.5,14,.95):self.detail((.48,.07,.004),(x,y,.006),'white',.0005)
        for x in [-13.2,13.2]:
            self.detail((.18,20,.018),(x,0,.008),'dark',.002)
            for y in np.arange(-9.9,10,.15):self.detail((.16,.028,.018),(x,y,.011),'metal',.001)
        for approach in d.get('bridge_approaches',[]):
            xa,xb=approach['x_range'];ya,yb=approach['y_range']
            for y in [ya,yb]:self.detail((xb-xa,.045,.004),((xa+xb)/2,y,.008),'paint',.0005)
            x=(xa+xb)/2;direction=1 if x<0 else -1
            mesh=self.UsdGeom.Mesh.Define(self.stage,'/World/Art/Arrow'+approach['name'])
            mesh.CreatePointsAttr([(x+direction*.6,0,.012),(x-direction*.25,-.4,.012),(x-direction*.25,.4,.012)])
            mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2]);mesh.CreateDoubleSidedAttr(True)
            mesh.CreateSubdivisionSchemeAttr('none');self.bind(mesh,'paint')
        for i,p in enumerate(d['targets']):
            self.painted_ring('/World/Art/Target'+str(i),p['center'],p['radius'],.045)
            self.sign('/World/Art/TargetLabel'+str(i),p['name'],'LOCATION / PREVIEW',(*p['center'],.015),.60,.25,'floor')
        for key,xy in d['spawns'].items():
            self.sign('/World/Art/Spawn'+key,'A / GO2-W' if key=='A' else 'B / G1','START / SCALE REFERENCE',
                (xy[0],xy[1]-1,.012),1.7,.42,'floor')
        self.sign('/World/Art/EntryPaint','DOCK 07','ROBOTICS PROVING GROUND / MAP REVIEW',(1.8,-10.7,.012),4.8,.8,'floor')

    def perimeter(self,d):
        hx,hy=np.array(d['playable_size_m'])/2
        for axis in ['x','y']:
            for side in [-1,1]:
                length=2*(hy if axis=='x' else hx);fixed=side*((hx if axis=='x' else hy)+.14)
                size=(.28,length,.95) if axis=='x' else (length,.28,.95)
                center=(fixed,0,.475) if axis=='x' else (0,fixed,.475)
                name=axis+('Neg' if side<0 else 'Pos')
                self.box('/World/Boundary/'+name,size,center,'weathered',.02,True)
                def pos(t,z):return (fixed,t,z) if axis=='x' else (t,fixed,z)
                for t in np.linspace(-length/2,length/2,int(length/2)+1):
                    self.detail((.085,.085,2.4),pos(t,1.2),'dark')
                wire=[]
                for t in np.arange(-length/2,length/2,.22):
                    delta=min(1.3,length/2-t)
                    wire.extend([[pos(t,1.02),pos(t+delta,1.02+delta)],
                                 [pos(t,2.32),pos(t+delta,2.32-delta)]])
                self.lines('/World/Art/Fence'+name,wire,'metal',.007)
                self.lines('/World/Art/Rail'+name,[[pos(-length/2,z),pos(length/2,z)] for z in [.99,2.35]],'metal',.045)
        # Large northern workshop and distant sheds close the low-camera horizon.
        for k,(x,y,sx,sy,h) in enumerate([(-2,17,29,7,5.4),(23,9,10,21,6),(-24,7,11,24,5.8),(3,-25,32,8,4.5)]):
            self.detail((sx,sy,h),(x,y,h/2),'concrete',.04)
            self.detail((sx+.3,sy+.3,.18),(x,y,h+.09),'dark')
            front=y-sy/2-.025
            for xx in np.arange(x-sx/2+.4,x+sx/2,1.8):
                self.detail((.085,.06,h-.2),(xx,front,h/2),'metal',.004)
            for xx in np.linspace(x-sx*.35,x+sx*.35,5):
                self.detail((2.6,.04,.65),(xx,front-.03,h-1.05),'glass',.008)
            if k==0:
                for xx in [-10,-3,4]:
                    self.detail((4.4,.11,3.2),(xx,front-.06,1.6),'blue')
                    for z in np.arange(.2,3.1,.20):self.detail((4.25,.025,.026),(xx,front-.13,z),'metal',.001)
                    self.sign('/World/Art/Bay'+str(int(xx+12)),'BAY '+str(int(xx+12)),'MAINTENANCE / KEEP CLEAR',
                        (xx,front-.14,3.55),3.6,.38)
                self.sign('/World/Art/MainSign','DOCK 07','AUTONOMOUS SYSTEMS / CONTAINER MAINTENANCE',
                    (-3,front-.15,4.75),10,.55)
                for xx in [-12,6,9]:
                    self.detail((1.7,1.8,.8),(xx,y,h+.5),'metal')
                    for z in np.arange(h+.2,h+.85,.09):self.detail((1.5,.025,.022),(xx,y-.92,z),'dark',.001)
        for x in [-15,15]:
            for y in [-10.8,10.8]:
                self.detail((.16,.16,6),(x,y,3),'metal')
                self.detail((1.2,.3,.12),(x,y,6),'dark')
                self.detail((1,.25,.018),(x,y,5.93),'lamp')
        self.lines('/World/Art/UtilityPipes',[[(-15.7,9,.45),(-15.7,11.7,.45),(15.5,11.7,.45)],
            [(-15.7,11.7,.65),(15.5,11.7,.65)]],'rust',.065)

    def build(self,d):
        self.ground(d)
        geometry=solids(d)
        for s in geometry:
            if s['shape']=='ramp':self.ramp(s)
            else:self.box('/World/Geometry/'+s['name'],s['size'],s['center'],s['material'],.012,True)
        self.walls(d);self.containers(d);self.bridge(d);self.perimeter(d)
        return dict(playable_size_m=d['playable_size_m'],structural_solids=len(geometry),
            assets=self.manifest,decorative_parts=self.serial,match_simulated=False,
            match_integration_ready=False,training=False,dynamics_validated=False,
            geometry_collision='Individual structural panels / ramps / deck / bridge rails / boundary. No solid tunnel proxy.',
            small_detail_collision=False,displacement_enabled=False,normal_maps='OpenGL',
            asset_paths='Absolute local paths; copy and repath dependencies for another computer.')
