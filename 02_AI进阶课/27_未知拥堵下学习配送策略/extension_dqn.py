"""离线拓展：更长路线中用同一环境比较 DQN、规则与已知模型参照。"""
from pathlib import Path
import os,time,json
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
torch.set_num_threads(2)
CASE_DIR=Path(__file__).resolve().parent
OUT=CASE_DIR/'outputs';OUT.mkdir(exist_ok=True)

class CourierEnv(gym.Env):
    def __init__(self):
        self.observation_space=spaces.Box(0,1,shape=(1,),dtype=np.float32)
        self.action_space=spaces.Discrete(2);self.state=20;self.steps=0
    def observation(self):return np.array([self.state/20],dtype=np.float32)
    def reset(self,seed=None,options=None):
        super().reset(seed=seed);self.state=20;self.steps=0
        return self.observation(),{}
    def step(self,action):
        self.steps+=1
        advance=int(action)==0 or self.np_random.random()<.58
        self.state=max(0,self.state-int(advance))
        reward=-2. if int(action)==0 else -1.
        return self.observation(),reward,self.state==0,self.steps>=200,{}

def evaluate(select_action,seeds):
    times=[];completed=[]
    for seed in seeds:
        env=CourierEnv();observation,_=env.reset(seed=seed);total=0.
        while True:
            action=int(select_action(observation));observation,reward,done,truncated,_=env.step(action);total-=reward
            if done or truncated:break
        times.append(total);completed.append(done)
    return {'mean_time':float(np.mean(times)),'completion_rate':float(np.mean(completed))}

if __name__=='__main__':
    check_env(CourierEnv(),warn=True)
    seeds=list(range(6000,6040));results=[]
    for training_seed in [1,2,3]:
        env=CourierEnv();started=time.perf_counter()
        model=DQN('MlpPolicy',env,policy_kwargs={'net_arch':[32,32]},learning_rate=.001,buffer_size=5000,learning_starts=100,batch_size=32,train_freq=4,gradient_steps=1,gamma=.99,exploration_fraction=.6,device='cpu',seed=training_seed,verbose=0)
        model.learn(total_timesteps=2500)
        elapsed=time.perf_counter()-started
        score=evaluate(lambda obs:model.predict(obs,deterministic=True)[0],seeds)
        results.append({'seed':training_seed,'training_steps':int(model.num_timesteps),'training_seconds':elapsed,**score})
    report={'environment':'20段路线；快路成功率0.58，慢路成本2','dqn':results,'safe_rule':evaluate(lambda obs:0,seeds),'known_model_reference':evaluate(lambda obs:1,seeds),'caveat':'三次训练仅用于小型教学比较；DQN不保证优于规则，主例5段路线的分数不能直接与这里比较。'}
    (OUT/'dqn_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
