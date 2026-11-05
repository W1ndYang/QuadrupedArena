"""v0.4.1 local patch; the v0.4 study source/records remain immutable.

When a peek is aborted while still moving out, brake along the current
heading before installing the opposite reference. This avoids abs(v)
advancing backwards before the signed velocity has actually changed sign.
"""
import math
import numpy as np
from sequence_skills import SequenceProbe as FrozenProbe
from peek_geometry import PEEK

class SequenceProbe(FrozenProbe):
    def execute(self,i,command,t):
        if i==0 and command['mode']=='PEEK_LOCAL':
            m=self.actor[0];delta=np.asarray(command['target'])-m.xy
            longitudinal=float(delta@np.array([math.cos(m.yaw),math.sin(m.yaw)]))
            if longitudinal*m.v < -1e-9 and abs(m.v)>1e-8:
                m.reverse_allowed=True;m.reverse_limit=PEEK['reverse_speed'];m.vmax=PEEK['speed'];m.accel=PEEK['acceleration']
                m.navigate([]);self.local_target=None;self.move_goal[0]=None
                return m.yaw,False,'PEEK_BRAKE'
        return super().execute(i,command,t)
