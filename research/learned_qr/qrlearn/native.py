"""Owned raw-RINEX sessions over the actual RTKLIB shared library.

Native pointers are borrowed during a callback. Copy before retaining. Python
exceptions are captured, then re-raised after C returns, never swallowed by ctypes.
"""
from __future__ import annotations
import ctypes as c
from contextlib import contextmanager
import os
from pathlib import Path
import threading
import numpy as np

D=c.POINTER(c.c_double)
I=c.POINTER(c.c_int)
class Event(c.Structure):
    _fields_=[('kind',c.c_int),('nx',c.c_int),('n',c.c_int),('m',c.c_int),
              ('time',c.c_double),('ix',I),('x',D),('P',D),('a',D),('b',D),('c',D)]
Callback=c.CFUNCTYPE(c.c_int,c.c_void_p,c.POINTER(Event))
MODES={'off':0,'q':1,'r':2,'qr':3}
_LOCK=threading.RLock()
_ENV=('RTKLIB_QR_STABLE','RTKLIB_QR_MODE','RTKLIB_QR_MODEL','RTKLIB_QR_LOG','RTKLIB_QR_ALLOW_TEST_MODEL')

@contextmanager
def native_call():
    # RTKLIB option parsing and some utilities are process-global. Also keep a
    # caller's deployment environment from altering a deliberately explicit session.
    with _LOCK:
        saved={k:os.environ.get(k) for k in _ENV}
        for k in _ENV:os.environ.pop(k,None)
        try:yield
        finally:
            for k,value in saved.items():
                if value is not None:os.environ[k]=value

def array(pointer,length:int)->np.ndarray:
    if length<0 or (length and not pointer):raise ValueError('invalid native array')
    return np.ctypeslib.as_array(pointer,shape=(length,)).copy() if length else np.empty(0)

def library_path(value=None)->Path:
    if value is not None:return Path(value).expanduser().resolve()
    root=Path(__file__).resolve().parents[3]
    for name in ('lib/librtklib.so','lib/librtklib.dylib','lib/rtklib.dll','bin/rtklib.dll'):
        if (root/name).exists():return root/name
    raise FileNotFoundError('Build first: cmake -S . -B build; cmake --build build --target rnx2rtkp')

class Session:
    def __init__(self,*,config,rover,base,nav,model=None,mode='off',training=True,
                 allow_test=False,max_gap=30.0,pair_age=0.05,library=None,stable=True,scl=False,candidates=False,observe_fixed=False,feature_log=None,pairing_latency=None):
        if feature_log is not None and Path(feature_log).exists():raise FileExistsError(feature_log)
        if mode not in MODES:raise ValueError('mode must be off/q/r/qr')
        if not stable and (training or mode!='off'):raise ValueError('legacy update is available only for non-learning replay')
        if not nav:raise ValueError('at least one navigation file is required')
        paths=[Path(p).expanduser().resolve() for p in (config,rover,base,*nav)]
        for p in paths:
            if not p.is_file():raise FileNotFoundError(p)
        if model is not None and not Path(model).is_file():raise FileNotFoundError(model)
        self.lib=c.CDLL(str(library_path(library)));self.ptr=None;self.mode=mode;self.model=model
        self.failed=False
        if not hasattr(self.lib,'qr_bridge_abi') or self.lib.qr_bridge_abi()!=1:
            raise RuntimeError('incompatible native library; rebuild this research branch')
        self.lib.qr_open.argtypes=[c.c_char_p,c.c_char_p,c.c_char_p,c.POINTER(c.c_char_p),
            c.c_int,c.c_char_p,c.c_int,c.c_int,c.c_int,c.c_double,c.c_double,c.c_char_p,c.c_int]
        self.lib.qr_open.restype=c.c_void_p
        self.lib.qr_step.argtypes=[c.c_void_p,Callback,c.c_void_p,c.c_char_p,c.c_int]
        self.lib.qr_step.restype=c.c_int
        self.lib.qr_restart.argtypes=[c.c_void_p,c.c_char_p,c.c_int]
        self.lib.qr_close.argtypes=[c.c_void_p]
        self.lib.qr_set_stable.argtypes=[c.c_void_p,c.c_int]
        navargs=(c.c_char_p*len(nav))(*(os.fsencode(p) for p in paths[3:]))
        error=c.create_string_buffer(1024)
        with native_call():
            self.ptr=self.lib.qr_open(*(os.fsencode(p) for p in paths[:3]),navargs,len(nav),
                os.fsencode(Path(model).resolve()) if model is not None else None,
                MODES[mode],int(training),int(allow_test),max_gap,pair_age,error,len(error))
        if not self.ptr:raise RuntimeError(error.value.decode('utf-8',errors='replace'))
        if not hasattr(self.lib,'qr_set_pairing_latency'):
            self.close();raise RuntimeError('rebuild native library for explicit pairing horizon')
        self.lib.qr_get_pairing_latency.argtypes=[c.c_void_p]
        self.lib.qr_get_pairing_latency.restype=c.c_double
        self.timestamp_horizon=float(self.lib.qr_get_pairing_latency(self.ptr) if pairing_latency is None else pairing_latency)
        self.lib.qr_set_pairing_latency.argtypes=[c.c_void_p,c.c_double]
        if self.lib.qr_set_pairing_latency(self.ptr,self.timestamp_horizon):
            self.close();raise ValueError('pairing latency must be finite, nonnegative, <= native DTTOL and <= pair_age')
        if scl or candidates or observe_fixed:
            if not hasattr(self.lib,'qr_set_learning_options'):
                self.close();raise RuntimeError('rebuild library with SCL observer')
            self.lib.qr_set_learning_options.argtypes=[c.c_void_p,c.c_int,c.c_int,c.c_int]
            if self.lib.qr_set_learning_options(self.ptr,2 if scl else 0,int(candidates),int(observe_fixed)):
                self.close();raise RuntimeError('invalid SCL options')
        if self.lib.qr_set_stable(self.ptr,int(stable)):
            self.close();raise RuntimeError('unsupported numerical update mode')
        if feature_log is not None:
            p=Path(feature_log).expanduser().resolve();p.parent.mkdir(parents=True,exist_ok=True)
            if not hasattr(self.lib,'qr_set_feature_log'):
                self.close();raise RuntimeError('rebuild native library for read-only feature logging')
            self.lib.qr_set_feature_log.argtypes=[c.c_void_p,c.c_char_p,c.c_char_p,c.c_int]
            if self.lib.qr_set_feature_log(self.ptr,os.fsencode(p),error,len(error)):
                self.close();raise RuntimeError(error.value.decode('utf-8',errors='replace'))

    def step(self,handler=None):
        if not self.ptr or self.failed:raise RuntimeError('session closed or failed; restart before reuse')
        if self.mode!='off' and self.model is None and handler is None:
            raise ValueError('learned mode needs an exported model or a training handler')
        output={};caught=[]
        def dispatch(_,pointer):
            if caught:return 1
            try:
                e=pointer.contents
                if handler is not None:handler(e)
                if e.kind==14:
                    meta=array(e.b,4)
                    output.update(time=e.time,status=e.n,ns=e.m,
                        position=array(e.a,6)[:3] if e.n else np.full(3,np.nan),
                        ratio=float(meta[0]),age=float(meta[1]),native_return=int(meta[2]),
                        release_t_gpst_s=e.time+max(0.,-float(meta[1])) if np.isfinite(meta[1]) else e.time,
                        timestamp_horizon_s=self.timestamp_horizon)
                return 0
            except BaseException as exc:
                caught.append(exc);return 1
        callback=Callback(dispatch);error=c.create_string_buffer(1024)
        with native_call():result=self.lib.qr_step(self.ptr,callback,None,error,len(error))
        if caught:
            self.failed=True;raise caught[0]
        if result<0:
            self.failed=True;raise RuntimeError(error.value.decode('utf-8',errors='replace'))
        if not result:return None
        if not output:raise RuntimeError('native session did not finish the epoch')
        return output

    def reload_model(self,path=None,*,allow_test=False,candidates=False):
        if not self.ptr or self.failed:raise RuntimeError('session not usable')
        self.lib.qr_reload_model.argtypes=[c.c_void_p,c.c_char_p,c.c_int,c.c_int,c.c_char_p,c.c_int]
        error=c.create_string_buffer(1024)
        with native_call():
            code=self.lib.qr_reload_model(self.ptr,os.fsencode(Path(path).resolve()) if path is not None else None,
                int(allow_test),int(candidates),error,len(error))
        if code:raise RuntimeError(error.value.decode())
        self.model=path

    def restart(self):
        if not self.ptr:raise RuntimeError('session closed')
        error=c.create_string_buffer(1024)
        with native_call():result=self.lib.qr_restart(self.ptr,error,len(error))
        if result:raise RuntimeError(error.value.decode('utf-8',errors='replace'))
        self.failed=False

    def close(self):
        if self.ptr:
            with native_call():self.lib.qr_close(self.ptr)
            self.ptr=None
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
