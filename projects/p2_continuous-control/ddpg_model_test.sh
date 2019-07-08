### Test for update_every
#python ddpg_experiment.py --save_replay --env Reacher_unity_v1 --batch_size 1028 --n_episode_bf_train 0 --lr_actor 1e-4 --lr_critic 5e-4 --num_episodes 400 --update_every 2 --actor_fc_units 128 --actor_fc_units2 64 --device gpu  --testname model_2_test3_update2 >  logs/ddpg_reacher_model_2_test3_update2
#python ddpg_experiment.py --save_replay --env Reacher_unity_v1 --batch_size 1028 --n_episode_bf_train 0 --lr_actor 1e-4 --lr_critic 5e-4 --num_episodes 400 --update_every 4 --actor_fc_units 128 --actor_fc_units2 64 --device gpu  --testname model_2_test3_update4 >  logs/ddpg_reacher_model_2_test3_update4
#python ddpg_experiment.py --save_replay --env Reacher_unity_v1 --batch_size 1028 --n_episode_bf_train 0 --lr_actor 1e-4 --lr_critic 5e-4 --num_episodes 400 --update_every 8 --actor_fc_units 128 --actor_fc_units2 64 --device gpu  --testname model_2_test3_update8 >  logs/ddpg_reacher_model_2_test3_update8
#python ddpg_experiment.py --save_replay --env Reacher_unity_v1 --batch_size 1028 --n_episode_bf_train 0 --lr_actor 1e-4 --lr_critic 5e-4 --num_episodes 400 --update_every 16 --actor_fc_units 128 --actor_fc_units2 64 --device gpu  --testname model_2_test3_update16 >  logs/ddpg_reacher_model_2_test3_update16

### Test for learning rate
#python ddpg_experiment.py --save_replay --env Reacher_unity_v1 --batch_size 1028 --n_episode_bf_train 0 --lr_actor 1e-4 --lr_critic 5e-4 --num_episodes 400 --update_every 4 --actor_fc_units 128 --actor_fc_units2 64 --device gpu  --testname model_2_test3_update4 >  logs/ddpg_reacher_model_2_test3_update4
#python ddpg_experiment.py --save_replay --env Reacher_unity_v1 --batch_size 1028 --n_episode_bf_train 0 --lr_actor 1e-4 --lr_critic 1e-3 --num_episodes 400 --update_every 4 --actor_fc_units 128 --actor_fc_units2 64 --device gpu  --testname ddpg_reacher_model_2_t3_lr_1e-3 >  logs/ddpg_reacher_model_2_t3_lr_1e-3

### Long run test
#python ddpg_experiment.py --save_replay --env Reacher_unity_v1 --batch_size 1028 --n_episode_bf_train 0 --lr_actor 1e-4 --lr_critic 1e-3 --num_episodes 5000 --update_every 4 --actor_fc_units 128 --actor_fc_units2 64 --device gpu  --testname ddpg_reacher_model_2_lr_1e-3 >  logs/ddpg_reacher_model_2_lr_1e-3

### Retrain 
python ddpg_experiment.py --save_replay --retrain --actor_weight models/actor_iter7200_Reacher_unity_v1_episodes_7200_score_29.0527993506193172019_06_26_12_38_14_ddpg_reacher_model_2_lr_1e-3_retart5600_checkpoint.pth  --critic_weight models/critic_iter7200_Reacher_unity_v1_episodes_7200_score_29.0527993506193172019_06_26_12_38_14_ddpg_reacher_model_2_lr_1e-3_retart5600_checkpoint.pth --replay_buffer_pickle models/Reacher_unity_v12019_06_26_12_38_14_replay_buffer_iter7200ddpg_reacher_model_2_lr_1e-3_retart5600 --restart_iter 7200 --env Reacher_unity_v1 --batch_size 1028 --n_episode_bf_train 0 --n_episode_stop_explore 0 --lr_actor 1e-5 --lr_critic 5e-5 --num_episodes 3000 --update_every 1 --actor_fc_units 128 --actor_fc_units2 64 --device gpu  --testname ddpg_reacher_model_2_lr_1e-3_retart7200 >  logs/ddpg_reacher_model_2_lr_1e-3_restart_7200


