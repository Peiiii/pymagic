import time,os,glob,shutil,random
import functools
class Config(dict):
    def __setattr__(self, key, value):
        self[key]=value
    def __getattr__(self, item):
        try:
            v=self[item]
            return v
        except:
            raise Exception('Config object has no key %s '%item)
    def add(self,key,value,description=''):
        self[key]=value
        self.description=description

class Timer:
    def __init__(self,verbose=False):
        self.history=[]
        self.dt_history = []
        self.steps=0
        self.start_time=time.time()
        self.history.append(self.start_time)
        self.verbose=verbose
        if self.verbose:
            print('Timer started at %s'%(self.start_time))

    def step(self):
        t=time.time()
        dt=t-self.history[-1]
        self.dt_history.append(dt)
        self.history.append(t)
        self.steps+=1

        if self.verbose:
            print('time since last step: %s'%(dt))
        return dt

    def end(self):
        t=time.time()
        self.end_time=t
        dt = t - self.history[-1]
        self.dt_history.append(dt)
        self.history.append(t)
        self.steps+=1
        if self.verbose:
            print('time since last step: %s'%(dt))
        return dt


class ListIterator:
    def __init__(self,mylist,batch_size):
        self.list=mylist
        self.batch_size=batch_size
        self.length=len(mylist)
        self.num_batch=self.length//self.batch_size
        self.cur_batch_idx=-1
        self.cur_batch=None
    def next(self):
        self.cur_batch_idx+=1
        if self.cur_batch_idx>=self.num_batch:
            self.cur_batch_idx=0
        self.cur_batch=self.get_batch_by_idx(self.cur_batch_idx)
        return self.cur_batch

    def get_batch_by_idx(self,batch_idx):
        st=batch_idx*self.batch_size
        ed=st+self.batch_size
        return self.list[st:ed]


def run_timer(func):
    name = func.__name__
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        print('running %s ...'%(name))
        t=Timer(verbose=False)
        ret=func(*args,**kwargs)
        dt=t.end()
        print('finished running %s ,time consumed: %s'%(name,dt))
        return ret
    return wrapper
