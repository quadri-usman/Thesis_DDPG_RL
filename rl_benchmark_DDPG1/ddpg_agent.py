"""DDPG agent for Stage-1 GFM PI-gain adaptation."""
from dataclasses import dataclass, asdict
from pathlib import Path
import random, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

@dataclass
class DDPGConfig:
    state_dim:int=8; action_dim:int=4
    actor_hidden:tuple=(256,256); critic_hidden:tuple=(256,256)
    actor_lr:float=1e-4; critic_lr:float=1e-3
    gamma:float=0.99; tau:float=5e-3
    buffer_size:int=500_000; batch_size:int=256; learning_starts:int=5_000
    exploration_std:float=0.20; exploration_std_min:float=0.02
    exploration_decay:float=0.99999; gradient_clip_norm:float=10.0; seed:int=0

class ReplayBuffer:
    def __init__(self,sd,ad,capacity,seed=0):
        self.capacity=capacity; self.rng=np.random.default_rng(seed)
        self.s=np.empty((capacity,sd),np.float32); self.a=np.empty((capacity,ad),np.float32)
        self.r=np.empty((capacity,1),np.float32); self.ns=np.empty((capacity,sd),np.float32)
        self.d=np.empty((capacity,1),np.float32); self.pos=0; self.size=0
    def __len__(self): return self.size
    def add(self,s,a,r,ns,d):
        i=self.pos; self.s[i]=s; self.a[i]=a; self.r[i]=r; self.ns[i]=ns; self.d[i]=d
        self.pos=(i+1)%self.capacity; self.size=min(self.size+1,self.capacity)
    def sample(self,n,device):
        i=self.rng.integers(0,self.size,n)
        T=lambda x: torch.as_tensor(x[i],dtype=torch.float32,device=device)
        return T(self.s),T(self.a),T(self.r),T(self.ns),T(self.d)

class Actor(nn.Module):
    def __init__(self,sd,ad,h=(256,256)):
        super().__init__(); layers=[]; n=sd
        for x in h: layers += [nn.Linear(n,x),nn.ReLU()]; n=x
        layers += [nn.Linear(n,ad),nn.Tanh()]; self.net=nn.Sequential(*layers)
    def forward(self,s): return self.net(s)

class Critic(nn.Module):
    def __init__(self,sd,ad,h=(256,256)):
        super().__init__(); layers=[]; n=sd+ad
        for x in h: layers += [nn.Linear(n,x),nn.ReLU()]; n=x
        layers += [nn.Linear(n,1)]; self.net=nn.Sequential(*layers)
    def forward(self,s,a): return self.net(torch.cat((s,a),-1))

class DDPGAgent:
    def __init__(self,cfg=None,device=None):
        self.cfg=cfg or DDPGConfig()
        if device is None: device="cuda" if torch.cuda.is_available() else "cpu"
        self.device=torch.device(device); c=self.cfg
        random.seed(c.seed); np.random.seed(c.seed); torch.manual_seed(c.seed)
        self.actor=Actor(c.state_dim,c.action_dim,c.actor_hidden).to(self.device)
        self.actor_target=Actor(c.state_dim,c.action_dim,c.actor_hidden).to(self.device)
        self.critic=Critic(c.state_dim,c.action_dim,c.critic_hidden).to(self.device)
        self.critic_target=Critic(c.state_dim,c.action_dim,c.critic_hidden).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.ao=optim.Adam(self.actor.parameters(),lr=c.actor_lr)
        self.co=optim.Adam(self.critic.parameters(),lr=c.critic_lr)
        self.replay=ReplayBuffer(c.state_dim,c.action_dim,c.buffer_size,c.seed)
        self.exploration_std=c.exploration_std; self.total_env_steps=0; self.total_updates=0
    @torch.no_grad()
    def select_action(self,state,explore=True):
        s=torch.as_tensor(np.asarray(state,np.float32).reshape(1,-1),device=self.device)
        a=self.actor(s).cpu().numpy()[0]
        if explore: a += np.random.normal(0,self.exploration_std,self.cfg.action_dim)
        return np.clip(a,-1,1).astype(np.float32)
    def remember(self,s,a,r,ns,done):
        self.replay.add(s,a,r,ns,done); self.total_env_steps+=1
    def ready(self):
        return len(self.replay)>=self.cfg.batch_size and self.total_env_steps>=self.cfg.learning_starts
    def update(self):
        if not self.ready(): return None
        c=self.cfg; s,a,r,ns,d=self.replay.sample(c.batch_size,self.device)
        with torch.no_grad():
            y=r+c.gamma*(1-d)*self.critic_target(ns,self.actor_target(ns))
        q=self.critic(s,a); cl=nn.functional.mse_loss(q,y)
        self.co.zero_grad(set_to_none=True); cl.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(),c.gradient_clip_norm); self.co.step()
        al=-self.critic(s,self.actor(s)).mean()
        self.ao.zero_grad(set_to_none=True); al.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(),c.gradient_clip_norm); self.ao.step()
        with torch.no_grad():
            for t,o in zip(self.actor_target.parameters(),self.actor.parameters()):
                t.mul_(1-c.tau).add_(o,alpha=c.tau)
            for t,o in zip(self.critic_target.parameters(),self.critic.parameters()):
                t.mul_(1-c.tau).add_(o,alpha=c.tau)
        self.total_updates+=1
        self.exploration_std=max(c.exploration_std_min,self.exploration_std*c.exploration_decay)
        return {"actor_loss":float(al.detach().cpu()),"critic_loss":float(cl.detach().cpu()),
                "q_mean":float(q.detach().mean().cpu()),"target_q_mean":float(y.detach().mean().cpu()),
                "exploration_std":float(self.exploration_std)}
    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"config":asdict(self.cfg),"actor":self.actor.state_dict(),
                    "actor_target":self.actor_target.state_dict(),"critic":self.critic.state_dict(),
                    "critic_target":self.critic_target.state_dict(),"ao":self.ao.state_dict(),
                    "co":self.co.state_dict(),"exploration_std":self.exploration_std,
                    "total_env_steps":self.total_env_steps,"total_updates":self.total_updates},path)
    def load(self,path,load_optimizers=True):
        x=torch.load(path,map_location=self.device)
        self.actor.load_state_dict(x["actor"]); self.actor_target.load_state_dict(x["actor_target"])
        self.critic.load_state_dict(x["critic"]); self.critic_target.load_state_dict(x["critic_target"])
        if load_optimizers: self.ao.load_state_dict(x["ao"]); self.co.load_state_dict(x["co"])
        self.exploration_std=float(x.get("exploration_std",self.cfg.exploration_std))
        self.total_env_steps=int(x.get("total_env_steps",0)); self.total_updates=int(x.get("total_updates",0))
    def parameter_counts(self):
        return {"actor":sum(p.numel() for p in self.actor.parameters()),
                "critic":sum(p.numel() for p in self.critic.parameters())}

if __name__=="__main__":
    a=DDPGAgent(); s=np.zeros(8,np.float32)
    print("device:",a.device); print("parameters:",a.parameter_counts())
    print("deterministic action:",a.select_action(s,False))
