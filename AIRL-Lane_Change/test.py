import gym
import time
import gymAutoDrive
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib import animation

env = gym.make('AutoDrive-v0')
anim_episode = 1
seed_l = list(range(anim_episode))
anim_steps = 301
is_visual = True  # Attention: if put it False, mush manually uncomment "yield" in anim_render.
is_draw = False

anim_time_start = time.time()
for k in range(anim_episode):
    ob = env.reset()
    selected_vehicle_ids = []
    figsize_tuple = (18, 9)
    fig, ax = plt.subplots(1, figsize=figsize_tuple)

    ob_vars = env.get_ob_vars(env.vehicle_list_separ_lanes)
    selected_vehicle_ids.append(env.lane_vehicle_id_from_env)

    for step in range(anim_steps):
        pending_acceleration, pending_yaw_acceleration = env.temp_agent_act(ob_vars)
        ob, ego_reward_tuple, ego_done = env.step(pending_acceleration, pending_yaw_acceleration)
        ob_vars = env.get_ob_vars(env.vehicle_list_separ_lanes)

        vehicle_patches_all, lane_vehicle_ids_all, vehicles_all, vehicle_ids_all = env.render(fig, ax)
        # Important!!  if set visual=False, must also manually uncomment yield, cannot use "if is_visual" as logical
        #vehicle_patches_all
        for patch in vehicle_patches_all:
            ax.add_patch(patch)
        ax.plot()
        plt.pause(0.001)
        #env._gather_data(k, step, vehicles_all)
        is_break = env._break_check(vehicles_all, k, step)

        if is_break or step > (anim_steps - 2):
            print(step, "------------------------", is_break)
            break


print("selected_vehicle_ids %s", selected_vehicle_ids)
anim_time_end = time.time()

print("evaluation time for one model %s" % (anim_time_end - anim_time_start))
print("------------------------------")
print("")
