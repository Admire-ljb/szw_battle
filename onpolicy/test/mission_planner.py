from traind_policy import *

class MissionPlanner:
    def __init__(self, env, policy_dic):
        self.env = env
        self.policy_dic = policy_dic

    def call_policy(self, policy_name, *args):
        self.policy_dic[policy_name].fly_run(*args)



if __name__ == "__main__":
    default_cfg = 'D:/crazyflie-simulation/airsim_mappo/onpolicy/envs/airsim_envs/cfg/default.cfg'
    cfg = Myconf()
    cfg.read(default_cfg)
    for each in cfg.items("algorithm"):
        cfg.__dict__[each[0]] = each[1]
    if cfg.getboolean('algorithm', 'cuda') and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(cfg.getint('algorithm', 'n_training_threads'))
        if cfg.getboolean('algorithm', 'cuda_deterministic'):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(cfg.getint('algorithm', 'n_training_threads'))

    # seed
    torch.manual_seed(cfg.getint('algorithm', 'seed'))
    torch.cuda.manual_seed_all(cfg.getint('algorithm', 'seed'))
    np.random.seed(cfg.getint('algorithm', 'seed'))

    # env init
    env = AirSimDroneEnv(cfg)
    num_agents = cfg.getint('options', 'num_of_drone')

    config = {
        "cfg": cfg,
        "envs": env,
        "num_agents": num_agents,
        "device": device
    }

    # load model
    policy_actor_state_dict = torch.load(str(cfg.get("algorithm", 'model_dir')) + '/actor.pt')
    actor1 = R_Actor(config['cfg'], config['envs'].observation_space[0], config['envs'].action_space[0], config['device'])
    actor1.load_state_dict(policy_actor_state_dict)

    patrol_drones = FixedPolicy("patrol_100.txt", ["127.0.0.1:41451", '10.134.142.129:41451'], 9699)
    patrol_drones.fly_run()
    attack_drones = AttackPolicy(actor1, env)
    mission_planner = MissionPlanner(env, {"attack": attack_drones})

    while True:
        attack_drones.attack_run(patrol_drones)

    # a.fly_run()
