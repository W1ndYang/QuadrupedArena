"""Native USD architecture for the approved A courtyard concept.

Structural boxes are exactly the referee's solids. Facade details are flush
skins (at most 12 mm), and ground markings introduce no movement obstacles.
Closed service doors and opaque windows never imply walkable interiors.
"""
import hashlib
import math
import numpy as np


def exposed_faces(cover, buildings):
    """Exterior intervals of a union of axis-aligned building footprints."""
    x,y=cover['center'];sx,sy,_=cover['size']
    for axis,side in ((0,-1),(0,1),(1,-1),(1,1)):
        fixed=(x,y)[axis]+side*(sx,sy)[axis]/2
        lo=(y,x)[axis]-(sy,sx)[axis]/2
        hi=(y,x)[axis]+(sy,sx)[axis]/2
        cuts={lo,hi}
        others=[b for b in buildings if b['name']!=cover['name']]
        for b in others:
            for edge in (-1,1):
                t=b['center'][1-axis]+edge*b['size'][1-axis]/2
                if lo<t<hi:cuts.add(t)
        cuts=sorted(cuts)
        for a,b in zip(cuts,cuts[1:]):
            xy=[0.,0.];xy[axis]=fixed+side*.002;xy[1-axis]=(a+b)/2
            hidden=any(all(abs(xy[k]-o['center'][k])<o['size'][k]/2-1e-6 for k in (0,1)) for o in others)
            if not hidden and b-a>.04:yield axis,side,fixed,a,b


def build(builder,cfg):
    b=builder
    b.material('yard_warm',(.64,.52,.40),texture='Concrete023',scale=.8)
    b.material('yard_cool',(.28,.39,.43),texture='Metal032',scale=1.,roughness=.77,metallic=.18)
    b.material('yard_pale',(.72,.73,.68),texture='Concrete034',scale=.9)
    b.material('yard_roof',(.20,.24,.26),texture='Metal032',scale=1.,roughness=.82,metallic=.18)
    b.material('yard_roof_light',(.49,.52,.51),texture='Concrete023',scale=.8)
    b.material('yard_slab',(.57,.57,.52),texture='Concrete034',scale=.75)
    b.material('yard_teal',(.09,.27,.30),roughness=.8,metallic=.12)
    b.material('yard_door',(.22,.27,.28),texture='Metal032',scale=1.3,roughness=.75,metallic=.18)
    b.material('yard_rubber',(.03,.035,.035),roughness=.95)
    for name in ['Metal032']:
        files=[b.assets/name/f'{name}_2K-JPG_{suffix}.jpg' for suffix in ('Color','NormalGL','Roughness')]
        b.manifest.append(dict(id=name,resolution='2K',license='CC0-1.0',
            source=f'https://ambientcg.com/view?id={name}',license_url='https://docs.ambientcg.com/license/',
            files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    b.box('/World/Floor',cfg['arena']['floor_size'],(0,0,-.1),'asphalt',.001,True)
    b.detail((100,100,.15),(0,0,-.22),'asphalt',.001)
    # Planar material inlays differentiate courtyards and service lanes.
    for x,y,sx,sy in [(-8.5,.1,7,6),(8.5,-1.3,7,6.2),(0,-3.8,24,3.0)]:
        b.detail((sx,sy,.002),(x,y,.001),'yard_slab',.0002)
    joints=[]
    for x in np.arange(-12,12.01,2):joints.append([(x,-8,.003),(x,8,.003)])
    for y in np.arange(-8,8.01,2):joints.append([(-12,y,.003),(12,y,.003)])
    b.lines('/World/Art/PavingJoints',joints,'joint',.008)
    buildings=[c for c in cfg['arena']['covers'] if c.get('architecture',{}).get('kind')=='building']
    facades=0
    for c in cfg['arena']['covers']:
        root='/World/Covers/'+c['name'];sx,sy,h=c['size'];x,y=c['center']
        art=c.get('architecture',{});kind=art.get('kind','low')
        mat=art.get('material','pale')
        b.box(root,c['size'],(x,y,h/2),mat,.018,True)
        if kind=='building':
            # All roof geometry stays inside the same bounding box and height.
            if min(sx,sy)>.5:
                b.detail((sx-.08,sy-.08,.035),(0,0,h/2-.020),'yard_roof_light' if mat=='yard_pale' else 'yard_roof',.004,root)
                for local_x in np.arange(-sx/2+.25,sx/2-.1,.75):
                    b.detail((.024,sy-.12,.018),(local_x,0,h/2-.009),'metal',.001,root)
            for axis,side,fixed,t0,t1 in exposed_faces(c,buildings):
                length=t1-t0;mid=(t0+t1)/2;facades+=1
                def point(t,z,depth=.004):
                    p=[0.,0.,z-h/2];p[axis]=fixed+side*depth-(x,y)[axis];p[1-axis]=t-(x,y)[1-axis];return tuple(p)
                def strip(width,height,t,z,material,depth=.004):
                    size=(.008,width,height) if axis==0 else (width,.008,height)
                    return b.detail(size,point(t,z,depth),material,.001,root)
                strip(length,.32,mid,.16,'weathered')
                strip(length,.10,mid,h-.07,'metal')
                strip(length,.11,mid,1.92,'yard_teal' if mat=='yard_cool' else 'paint')
                for t in np.arange(t0+.1,t1-.04,1.25):
                    strip(.014,1.55,t,1.1,'joint',.006)
                # Only long facade intervals receive doors, glazing and labels.
                if length>2.0 and art.get('label'):
                    door_t=t0+min(.95,length*.28)
                    strip(min(1.25,length*.38),1.55,door_t,1.04,'yard_door',.008)
                    for z in np.arange(.35,1.78,.13):strip(min(1.17,length*.36),.018,door_t,z,'metal',.010)
                    strip(.07,.18,door_t+.40,1.0,'dark',.012)
                    if length>3.8:
                        win_t=t1-min(1.05,length*.22)
                        strip(min(1.4,length*.26),.49,win_t,1.42,'dark',.007)
                        strip(min(1.30,length*.24),.39,win_t,1.42,'glass',.009)
                        strip(.04,.43,win_t,1.42,'metal',.011)
                    label_width=min(2.4,length-.25)
                    label_pos=point(mid,2.16,.011)
                    sign_path=root+f'/FacadeID{facades}'
                    b.sign(sign_path,art['label'],art.get('subtitle','SERVICE / CLOSED'),
                           label_pos,label_width,.30,'wall_x' if axis==0 else 'wall',
                           color=(132,188,190) if mat=='yard_cool' else (210,175,108))
                    if side>0:
                        # The base label quad faces -X/-Y. Positive faces need
                        # reversed U coordinates so lettering reads from outside.
                        uv=b.UsdGeom.PrimvarsAPI(b.stage.GetPrimAtPath(sign_path)).GetPrimvar('st')
                        uv.Set([(1.,0.),(0.,0.),(0.,1.),(1.,1.)])
        else:
            # Small cover is visibly equipment protection, not a miniature house.
            longx=sx>=sy;length=max(sx,sy)
            b.detail((sx-.008,sy-.008,.045),(0,0,h/2-.025),'metal',.004,root)
            b.detail((sx+.004,sy+.004,.075),(0,0,-h/2+.12),'paint',.002,root)
            for fraction in [-.36,.36]:
                for side in [-1,1]:
                    pos=(fraction*sx,side*(sy/2+.004),0) if longx else (side*(sx/2+.004),fraction*sy,0)
                    size=(.035,.008,min(.24,h*.55)) if longx else (.008,.035,min(.24,h*.55))
                    b.detail(size,pos,'dark',.002,root)
            if kind=='collapse':
                pos=(0,-sy/2-.006,.40) if longx else (-sx/2-.006,0,.40)
                b.sign(root+'/CollapseID','C-20','REPLACEABLE PANEL',pos,min(1.4,length*.8),.30,'wall' if longx else 'wall_x')
                for z in (-.55,.15):b.detail((sx+.004,sy+.004,.025),(0,0,z),'joint',.001,root)
    if cfg['collapse'].get('debris_pieces'):
        # Each piece has its own exact collider. The middle gap stays empty.
        root='/World/Rubble/'+cfg['collapse']['wall']
        b.UsdGeom.Xform.Define(b.stage,root)
        rng=np.random.default_rng(2902)
        for debris in cfg['collapse']['debris_pieces']:
            sx,sy,h=debris['size'];child=root+'/'+debris['name']
            b.box(child,debris['size'],(*debris['center'],h/2),'weathered',.015,True)
            if h<.5:
                b.UsdGeom.Imageable(b.stage.GetPrimAtPath(child+'/Visual')).MakeInvisible()
                b.detail((sx-.02,sy-.02,.065),(0,0,-h/2+.0325),'weathered',.012,child)
                for index in range(32):
                    wx=float(rng.uniform(.18,.44));wy=float(rng.uniform(.14,.35));radius=float(np.hypot(wx,wy)/2)
                    x=float(rng.uniform(-sx/2+radius,sx/2-radius));y=float(rng.uniform(-sy/2+radius,sy/2-radius))
                    height=h-.065 if index==0 else float(rng.uniform(.07,h-.1))
                    piece=b.detail((wx,wy,height),(x,y,-h/2+.065+height/2),'weathered',.012,child)
                    piece.AddRotateZOp().Set(float(rng.uniform(-180,180)))
        b.UsdGeom.Imageable(b.stage.GetPrimAtPath(root)).MakeInvisible()
    elif cfg['collapse'].get('mode')=='rubble':
        # Conservative solid footprint shared with the referee, with broken
        # concrete slabs entirely inside its bounds and maximum height.
        debris=cfg['collapse']['debris'];sx,sy,h=debris['size']
        root='/World/Rubble/'+cfg['collapse']['wall']
        b.box(root,debris['size'],(*debris['center'],h/2),'weathered',.015,True)
        b.UsdGeom.Imageable(b.stage.GetPrimAtPath(root+'/Visual')).MakeInvisible()
        b.detail((sx-.02,sy-.02,.065),(0,0,-h/2+.0325),'weathered',.012,root)
        rng=np.random.default_rng(2902)
        for index in range(42):
            wx,wy=rng.uniform(.28,.78,size=2);radius=float(np.hypot(wx,wy)/2)
            x=float(rng.uniform(-sx/2+radius,sx/2-radius))
            y=float(rng.uniform(-sy/2+radius,sy/2-radius))
            height=.285 if index==0 else float(rng.uniform(.07,.21))
            bottom=.065 if index==0 else float(rng.uniform(.04,.12))
            piece=b.detail((float(wx),float(wy),height),(x,y,-h/2+bottom+height/2),
                           'yard_cool' if index%6==0 else 'weathered',.018,root)
            piece.AddRotateZOp().Set(float(rng.uniform(-180,180)))
        b.UsdGeom.Imageable(b.stage.GetPrimAtPath(root)).MakeInvisible()
    # Realistic perimeter outside the legal arena. No additional internal props.
    for x in (-12.35,12.35):
        b.detail((.15,16.4,.25),(x,0,.125),'weathered')
        for y in np.arange(-8,8.1,2):b.detail((.055,.055,2),(x,y,1),'metal',.002)
        b.lines(f'/World/Art/Perimeter{int(x+13)}',[[(x,-8,z),(x,8,z)] for z in (.3,1.05,1.95)],'metal',.023)
    for y in (-8.28,8.28):
        b.detail((24.4,.12,.20),(0,y,.10),'weathered')
    # Painted edge only; these markings do not change traversability.
    for y in (-7.96,7.96):b.detail((23.92,.035,.002),(0,y,.004),'paint',.0002)
    for x in (-11.96,11.96):b.detail((.035,15.92,.002),(x,0,.004),'paint',.0002)
    for i,p in enumerate(cfg['arena']['task_points']):
        b.painted_ring('/World/Art/Target'+str(i),p['center'],p['radius'],.055)
        b.sign('/World/Art/TargetLabel'+str(i),p['name'],'CONTROL POINT',(*p['center'],.010),.57,.25,'floor')
    b.sign('/World/Art/WestYard','WEST / 01','SERVICE COURT',(-9.6,-2.9,.008),1.5,.4,'floor')
    b.sign('/World/Art/EastYard','EAST / 02','LOADING COURT',(9.5,-2.7,.008),1.5,.4,'floor',color=(132,188,190))
    b.sign('/World/Art/Entry','YARD A','ROBOTICS TEST FACILITY',(-6.4,-7.65,.009),2.4,.40,'floor')
    return dict(playable_size_m=cfg['arena']['floor_size'][:2],architecture='plan_a_offset_courtyards',
        gameplay_changed=True,structural_boxes=len(cfg['arena']['covers']),building_union_facade_intervals=facades,
        structural_geometry='Exact referee AABBs; closed buildings, floor-height paths; no traversable interiors',
        decorative_collision=False,facade_skin_max_offset_m=.012,roof_details_within_structural_height=True,
        assets=b.manifest,decorative_parts=b.serial,displacement_enabled=False,
        normal_maps='OpenGL, flip_tangent_v=False',concept_layout_preserved=not cfg['arena'].get('concept',{}).get('layout_changed',False))
