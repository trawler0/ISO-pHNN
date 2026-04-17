# noise
python main.py --name motor --num_trajectories 16 --num_val_trajectories 100 --hidden_dim 16 --lr 1e-3 --epochs 4000 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name no_noise --experiment noise_f
python main.py --name motor --num_trajectories 16 --num_val_trajectories 100 --hidden_dim 16 --lr 1e-3 --epochs 4000 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name noise20 --experiment noise_f --dB 20
python main.py --name motor --num_trajectories 16 --num_val_trajectories 100 --hidden_dim 16 --lr 1e-3 --epochs 4000 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name noise25 --experiment noise_f --dB 25
python main.py --name motor --num_trajectories 16 --num_val_trajectories 100 --hidden_dim 16 --lr 1e-3 --epochs 4000 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name noise30 --experiment noise_f --dB 30
python main.py --name motor --num_trajectories 16 --num_val_trajectories 100 --hidden_dim 16 --lr 1e-3 --epochs 4000 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name noise35 --experiment noise_f --dB 35
  