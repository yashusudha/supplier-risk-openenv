from fastapi import FastAPI
from supplier_env.env import SupplierRiskEnv
from supplier_env.tasks import get_task
from supplier_env.models import Action

app = FastAPI()

env = SupplierRiskEnv(get_task("easy"), seed=42)
env.reset()


@app.get("/")
def root():
    return {"status": "ok", "message": "Supplier Risk OpenEnv running"}


# @app.post("/reset")
# def reset(task: str = "easy"):
#     global env
#     env = SupplierRiskEnv(get_task(task), seed=42)
#     obs = env.reset()
#     return obs.model_dump()
@app.get("/reset")
@app.post("/reset")
def reset(task: str = "easy"):
    obs = env.reset(task=task)
    return obs.model_dump()


@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state()