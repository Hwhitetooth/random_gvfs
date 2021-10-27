from ray import tune

from algorithms.a2c_rgvfs import Experiment


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop_gradient', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = {
        'label': 'rgvfs',

        'env_id': tune.grid_search([
            'dmlab/contributed/dmlab30/explore_object_locations_small',
            'dmlab/contributed/dmlab30/explore_object_locations_large',
            'dmlab/contributed/dmlab30/explore_obstructed_goals_small',
            'dmlab/contributed/dmlab30/explore_obstructed_goals_large',
            'dmlab/contributed/dmlab30/explore_goal_locations_small',
            'dmlab/contributed/dmlab30/explore_goal_locations_large',
            'dmlab/contributed/dmlab30/explore_object_rewards_few',
            'dmlab/contributed/dmlab30/explore_object_rewards_many',
            'dmlab/contributed/dmlab30/lasertag_one_opponent_small',
            'dmlab/contributed/dmlab30/lasertag_one_opponent_large',
            'dmlab/contributed/dmlab30/lasertag_three_opponents_small',
            'dmlab/contributed/dmlab30/lasertag_three_opponents_large',
        ]),
        'cache': 'local',
        'noop_max': 30,
        'env_kwargs': {
            'width': 96,
            'height': 72,
        },

        'torso_type': 'atari_shallow',
        'torso_kwargs': {
            'dense_layers': (512,),
        },
        'use_rnn': True,
        'head_layers': (512,),
        'stop_ac_grad': args.stop_gradient,

        'nenvs': 32,
        'nsteps': 20,
        'gamma': 0.99,
        'lambda_': 1.,
        'vf_coef': 0.5,
        'entropy_reg': 0.003,

        'a2c_opt_kwargs': {
            'learning_rate': 3E-4,
            'decay': 0.99,
            'eps': 1E-8,
        },
        'max_a2c_grad_norm': 0.5,

        'aux_opt_kwargs': {
            'learning_rate': 3E-4,
            'b1': 0.,
            'b2': 0.99,
            'eps_root': 1E-8,
        },
        'max_aux_grad_norm': 0.,

        'td_net_kwargs': {
            'seed': None,
            'depth': 8,
            'repeat': 16,
            'discount_factors': (0.95,),
        },

        'random_feature_kwargs': {
            'conv_layers': ((1, (18, 24), (18, 24)),),
            'dense_layers': (),
            'padding': 'VALID',
            'w_init': 'orthogonal',
            'w_init_scale': 1.,
            'delta': True,
            'absolute': True,
            'only_last_channel': False,
        },

        'log_interval': 100,
        'seed': args.seed,
    }
    analysis = tune.run(
        Experiment,
        name='dmlab',
        config=config,
        stop={
            'num_frames': 200 * 10 ** 6,
        },
        resources_per_trial={
            'gpu': 1,
        },
    )
