from supplier_env.env import SupplierRiskEnv
from supplier_env.tasks import get_task
from supplier_env.models import Action
from supplier_env.grader import grade_episode

task = get_task("easy")
env = SupplierRiskEnv(task, seed=42)

obs = env.reset()
done = False

while not done:
    action = Action(order_A=50, order_B=30, order_C=40)  # dummy policy
    obs, reward, done, info = env.step(action)

print("Episode finished")
print("Final score:", grade_episode(env))