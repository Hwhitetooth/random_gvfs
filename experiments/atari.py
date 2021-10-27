from ray import tune

from algorithms.a2c_rgvfs import Experiment


GAME_LIST = [
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "Asteroids",
    "Atlantis",
    "BankHeist",
    "BattleZone",
    "BeamRider",
    "Bowling",
    "Boxing",
    "Breakout",
    "Centipede",
    "ChopperCommand",
    "CrazyClimber",
    "DemonAttack",
    "DoubleDunk",
    "Enduro",
    "FishingDerby",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Gravitar",
    "Hero",
    "IceHockey",
    "Jamesbond",
    "Kangaroo",
    "Krull",
    "KungFuMaster",
    "MontezumaRevenge",
    "MsPacman",
    "NameThisGame",
    "Pong",
    "PrivateEye",
    "Qbert",
    "Riverraid",
    "RoadRunner",
    "Robotank",
    "Seaquest",
    "SpaceInvaders",
    "StarGunner",
    "Tennis",
    "TimePilot",
    "Tutankham",
    "UpNDown",
    "Venture",
    "VideoPinball",
    "WizardOfWor",
    "Zaxxon",
]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop_gradient', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = {
        'label': 'rgvfs',

        'env_id': tune.grid_search([
            game + 'NoFrameskip-v4'
            for game in GAME_LIST
        ]),
        'env_kwargs': {},

        'torso_kwargs': {
            'dense_layers': (),
        },
        'use_rnn': False,
        'head_layers': (512,),
        'stop_ac_grad': args.stop_gradient,

        'nenvs': 16,
        'nsteps': 20,
        'gamma': 0.99,
        'lambda_': 1.,
        'vf_coef': 0.5,
        'entropy_reg': 0.01,

        'a2c_opt_kwargs': {
            'learning_rate': 7E-4,
            'decay': 0.99,
            'eps': 1E-5,
        },
        'max_a2c_grad_norm': 0.5,

        'aux_opt_kwargs': {
            'learning_rate': 7E-4,
            'b1': 0.,
            'b2': 0.99,
            'eps_root': 1E-5,
        },
        'max_aux_grad_norm': 0.,

        'td_net_kwargs': {
            'seed': None,
            'depth': 8,
            'repeat': 16,
            'discount_factors': (0.95,),
        },

        'random_feature_kwargs': {
            'conv_layers': ((1, 21, 21),),
            'dense_layers': (),
            'padding': 'VALID',
            'w_init': 'orthogonal',
            'w_init_scale': 8.,
            'delta': True,
            'absolute': True,
            'only_last_channel': True,
        },

        'log_interval': 100,
        'seed': args.seed,
    }
    analysis = tune.run(
        Experiment,
        name='atari',
        config=config,
        stop={
            'num_frames': 200 * 10 ** 6,
        },
        resources_per_trial={
            'gpu': 1,
        },
    )
