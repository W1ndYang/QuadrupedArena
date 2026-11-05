"""Depth-tested native USD stun marker, shared by live RGB and replay.

No screen-space indicator, wall bypass, collider, detector or planner telemetry.
"""
import math


class StunStars:
    def __init__(self,stage):
        from pxr import UsdGeom,UsdShade,Sdf,Gf
        self.Gf=Gf
        self.root=UsdGeom.Xform.Define(stage,'/World/StunStars')
        self.move=self.root.AddTranslateOp();self.spin=self.root.AddRotateZOp()
        material=UsdShade.Material.Define(stage,'/World/StunStarsMaterial')
        shader=UsdShade.Shader.Define(stage,'/World/StunStarsMaterial/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.,.7,.025))
        shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.8,.4,.01))
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.35)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        for j in range(3):
            angle=j*2*math.pi/3
            item=UsdGeom.Mesh.Define(stage,f'/World/StunStars/Star{j}')
            # Extruded vertical five-point star; orbital radius .27 m.
            vertices=[]
            for depth in (-.018,.018):
                for i in range(10):
                    a=math.pi/2+i*math.pi/5;r=.12 if i%2==0 else .05
                    x=r*math.cos(a);z=r*math.sin(a)
                    vertices.append((.27*math.cos(angle)+x*math.cos(angle)-depth*math.sin(angle),
                                     .27*math.sin(angle)+x*math.sin(angle)+depth*math.cos(angle),z))
            vertices.extend([(.27*math.cos(angle)+d*-math.sin(angle),.27*math.sin(angle)+d*math.cos(angle),0.) for d in (-.018,.018)])
            faces=[]
            for i in range(10):
                k=(i+1)%10
                faces.extend([[20,k,i],[21,i+10,k+10],[i,k,k+10],[i,k+10,i+10]])
            item.CreatePointsAttr(vertices);item.CreateFaceVertexCountsAttr([3]*len(faces))
            item.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
            item.CreateSubdivisionSchemeAttr('none');item.CreateDoubleSidedAttr(True)
            UsdShade.MaterialBindingAPI.Apply(item.GetPrim()).Bind(material)
        self.root.MakeInvisible()

    def update(self,state):
        if state.get('stun_remaining',[0,0])[1]>0 and state['hp'][1]>0:
            self.root.MakeVisible()
            self.move.Set(self.Gf.Vec3d(*state['xy'][1],1.58))
            self.spin.Set(float(state['t']*160%360))
        else:self.root.MakeInvisible()
