"""No-training SD variance rules; native DD construction keeps shared-reference terms.

Huber rule is one-pass predicted-innovation IRLS (not iterative robust RTK):
weight=min(1,k/|z|), so the variance multiplier is max(1,|z|/k).
Only code is adapted, like the SCL model; phase and original AR are untouched.
"""
import math
import numpy as np
from .native import array


def variance_multiplier(history,method,*,gain=10.,k=1.5,threshold=2.,width=4.,cap=1000.):
    h=np.asarray(history,dtype=float)
    if h.shape!=(10,13) or not np.isfinite(h).all():raise ValueError('expected finite V3 history10x13')
    if method not in ('fixed','huber'):raise ValueError('unknown variance rule')
    if not all(math.isfinite(v) for v in (gain,k,threshold,width,cap)) or not 1<=gain<=cap or k<=0 or width<=0 or cap<1:
        raise ValueError('invalid rule parameters')
    if h[-1,-1]<=.5 or h[-1,3]<=.5:return 1.
    z=math.expm1(min(8.,abs(h[-1,8])))
    if method=='huber':return min(cap,max(1.,z/k))
    evidence=max(z,math.expm1(min(8.,abs(h[-1,9]))))
    gate=min(1.,max(0.,(evidence-threshold)/width))
    return 1.+gate*(gain-1.)


class RuleController:
    def __init__(self,method,**parameters):
        self.method=method;self.parameters=parameters;self.calls=0;self.inflated=0
    def handle(self,event):
        if event.kind!=6:return
        h=array(event.a,10*13).reshape(10,13)
        scale=variance_multiplier(h,self.method,**self.parameters)
        event.b[0]=scale;self.calls+=1;self.inflated+=int(scale>1.)
