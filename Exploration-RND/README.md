### Usage:
To reproduce our results:

#### Baseline DQN
```
python3 main.py --gpu 0 --gamma 0.98 --env_name MiniGrid-FourRooms-7x7-v0 --max_game_len 99 --mode DQNm --mn 0.0 --buffer_size 1000000 --max_epoch 100000
```

#### DQN - 0.5

```
python3 main.py --gpu 0 --gamma 0.98 --env_name MiniGrid-FourRooms-7x7-v0 --max_game_len 99 --mode DQNm --mn 0.5 --buffer_size 1000000 --max_epoch 100000
```

#### Vanilla RND

```
python3 main.py --gpu 5 --gamma 0.98 --env_name MiniGrid-FourRooms-7x7-v0 --max_game_len 99 --mode RNDm --mn 1.5 --buffer_size 1000000 --max_epoch 100000
```

#### RND -1.0

```
python3 main.py --gpu 5 --gamma 0.98 --env_name MiniGrid-FourRooms-7x7-v0 --max_game_len 99 --mode RNDm --mn 1.0 --buffer_size 1000000 --max_epoch 100000
```

#### RND with Reward Shifting Exploration: RND -1.5

```
python3 main.py --gpu 5 --gamma 0.98 --env_name MiniGrid-FourRooms-7x7-v0 --max_game_len 99 --mode RNDm --mn 1.5 --buffer_size 1000000 --max_epoch 100000
```