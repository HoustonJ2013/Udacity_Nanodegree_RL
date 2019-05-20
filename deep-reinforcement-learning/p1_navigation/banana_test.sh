python agent_experiment.py --device gpu --env Banana_unity --max_t 500 --batch_size 64 --num_episodes 2000 --score_threshold 20 --score_window_size 100 --update_every 10 --eps_decay 0.998 --testname banana_reg_20 --buffer_size 10000 

python agent_experiment.py --device gpu --env Banana_unity --max_t 500 --batch_size 64 --num_episodes 2000 --score_threshold 20 --score_window_size 100 --update_every 10 --eps_decay 0.998 --testname banana_per_20 --buffer_size 10000 --per
