import uuid,json
WORD_DIC_PATH='words.wdic'
LIST_DIC_PATH='lists.ldic'


class Word(dict):
    def __init__(self,spell,meanings=[],id=None,fathers=None):
        self.spell=spell
        self.meanings=meanings
        self.id=uuid.uuid4().hex if not id else id
        self.fathers=set() if not fathers else set(fathers)
    def __getattr__(self, item):
        try:
            v=self[item]
            return v
        except:
            print('Word has no attribute %s'%(item))
    def __setattr__(self, key, value):
        self[key]=value
    def to_json(self):
        return self
    @classmethod
    def from_json(cls,dic):
        w=cls(**dic)
        return w
    def show(self,D):
        tmpW=self.copy()
        tmp_fathers=[]
        for l in tmpW['fathers']:
            tmp_fathers.append(D[l])
        tmpW['fathers']=tmp_fathers
        print(tmpW)

class WList(list):
    def __init__(self,words=[],id=None,description=''):
        self.id = uuid.uuid4().hex if not id else id
        self.description=description
        super().__init__(words)
    def update(self,dic):
        for w in self:
            try:
                w = dic[w]
                w.fathers.add(self.id)
            except:
                print('warning: list word "%s" not exists in dic'%(w))
                continue
    def to_json(self):
        return self
    @classmethod
    def from_json(cls,li):
        w=cls(li)
        return w

cls_dic={
            'word':Word,
            'wlist':WList
        }

class Dictionary(dict):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    def add(self,k,v):
        self[k]=v
    def save(self,fp):
        tmpD={}
        for k,v in self.items():
            tmpD[k]=v.to_json()
        with open(fp,'w') as f:
            json.dump(tmpD,f)
        print('Save dictionary to %s .'%(fp))
    @classmethod
    def load(cls,fp,item_cls):
        with open(fp,'r') as f:
            tmpD=json.load(f)
        for k,v in tmpD.items():
            tmpD[k]=item_cls.from_json(v)
        D=cls(tmpD)
        print('Load dictionary from %s'%(fp))
        return D






w1=Word(spell='stall',meanings=['摊位'])
w2=Word(spell='mourn',meanings=['悼念'])

# print(w1.keys())

WD=Dictionary()
WD.add(w1.spell,w1)
WD.add(w2.spell,w2)

l1=WList(['stall','mourn'])
l1.update(WD)
LD=Dictionary()
LD.add(l1.id,l1)
w1.show(LD)