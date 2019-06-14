python ddpg_experiment.py --env Reacher_unity_v1 --batch_size 1028 --n_episode_bf_train 0 --lr_actor 1e-4 --lr_critic 3e-4 --num_episodes 200 --update_every 1 --actor_fc_units 256 --critic_fc_units1 256 --critic_fc_units2 256 --critic_fc_units3 128 --device gpu  >  logs/ddpg_reacher_model_t3

python ddpg_experiment.py --env Reacher_unity_v1 --batch_size 1028 --n_episode_bf_train 0 --lr_actor 1e-4 --lr_critic 5e-4 --num_episodes 3000 --update_every 1 --device gpu  >  logs/ddpg_reacher_model_benchmark_3000
