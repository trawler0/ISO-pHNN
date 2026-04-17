
for NUM_TRAJECTORIES in 4 8 16 32 64; do
  for EPOCHS in 500 1000 2000 4000; do
    for HIDDEN_DIM in  4 6 8 10 12 14 16 18 20 22 24 26; do
        HIDDEN_KAN=$(( (HIDDEN_DIM + 4 - 1) / 4 ))
        EPOCHS_KAN=$(( EPOCHS / 4))
        echo "$HIDDEN_DIM" "$HIDDEN_KAN"
        seed=$(( HIDDEN_DIM+EPOCHS+NUM_TRAJECTORIES ))
        sbatch scripts/cluster_baseline.sh  "motor" "$seed" "data/motor_${NUM_TRAJECTORIES}.npy" "$NUM_TRAJECTORIES" "$EPOCHS" $HIDDEN_DIM 0.001 none none none none scaling_motor_paper motor_baseline_$NUM_TRAJECTORIES
        sbatch scripts/cluster.sh  "motor" "$seed" "data/motor_${NUM_TRAJECTORIES}.npy" "$NUM_TRAJECTORIES" "$EPOCHS" $HIDDEN_DIM 0.001 default default mlp gradient_positive scaling_motor_paper motor_default_$NUM_TRAJECTORIES
        sbatch scripts/cluster_no_norm.sh  "motor" "$seed" "data/motor_${NUM_TRAJECTORIES}.npy" "$NUM_TRAJECTORIES" "$EPOCHS" $HIDDEN_DIM 0.001 default linear linear linear scaling_motor_paper motor_prior_$NUM_TRAJECTORIES
        sbatch scripts/cluster.sh  "motor" "$seed" "data/motor_${NUM_TRAJECTORIES}.npy" "$NUM_TRAJECTORIES" "$EPOCHS_KAN" $HIDDEN_KAN 0.001 default_kan default_kan kan gradient_kan_positive scaling_motor_paper motor_kan_$NUM_TRAJECTORIES

        sbatch scripts/cluster_baseline.sh  "spring" "$seed" "data/spring_${NUM_TRAJECTORIES}.npy" "$NUM_TRAJECTORIES" "$EPOCHS" $HIDDEN_DIM 0.001 none none none none scaling_spring_paper spring_baseline_$NUM_TRAJECTORIES
        sbatch scripts/cluster.sh  "spring" "$seed" "data/spring_${NUM_TRAJECTORIES}.npy" "$NUM_TRAJECTORIES" "$EPOCHS" $HIDDEN_DIM 0.001 default default mlp gradient_positive scaling_spring_paper spring_default_$NUM_TRAJECTORIES
        sbatch scripts/cluster_no_norm.sh  "spring" "$seed" "data/spring_${NUM_TRAJECTORIES}.npy" "$NUM_TRAJECTORIES" "$EPOCHS" $HIDDEN_DIM 0.001 linear default linear linear scaling_spring_paper spring_prior_$NUM_TRAJECTORIES
        sbatch scripts/cluster.sh  "spring" "$seed" "data/spring_${NUM_TRAJECTORIES}.npy" "$NUM_TRAJECTORIES" "$EPOCHS_KAN" $HIDDEN_KAN 0.001 default_kan default_kan kan gradient_kan_positive scaling_spring_paper spring_kan_$NUM_TRAJECTORIES

        sbatch scripts/cluster_baseline.sh  "ball" "$seed" "data/ball_${NUM_TRAJECTORIES}.npy" "$NUM_TRAJECTORIES" "$EPOCHS" $HIDDEN_DIM 0.001 none none none none scaling_ball_paper ball_baseline_$NUM_TRAJECTORIES
        sbatch scripts/cluster.sh  "ball" "$seed" "data/ball_${NUM_TRAJECTORIES}.npy" "$NUM_TRAJECTORIES" "$EPOCHS" $HIDDEN_DIM 0.001 default default mlp gradient_positive scaling_ball_paper ball_default_$NUM_TRAJECTORIES
        sbatch scripts/cluster_no_norm.sh  "ball" "$seed" "data/ball_${NUM_TRAJECTORIES}.npy" "$NUM_TRAJECTORIES" "$EPOCHS" $HIDDEN_DIM 0.001 linear default linear gradient_positive scaling_ball_paper ball_prior_$NUM_TRAJECTORIES
        sbatch scripts/cluster.sh  "ball" "$seed" "data/ball_${NUM_TRAJECTORIES}.npy" "$NUM_TRAJECTORIES" "$EPOCHS_KAN" $HIDDEN_KAN 0.001 default_kan default_kan kan gradient_kan_positive scaling_ball_paper ball_kan_$NUM_TRAJECTORIES

        sbatch scripts/cluster_no_norm.sh  "spring" "$seed" "data/spring_${NUM_TRAJECTORIES}.npy" "$NUM_TRAJECTORIES" "$EPOCHS" $HIDDEN_DIM 0.001 linear linear linear linear prior_comparison_paper wrong
        sbatch scripts/cluster_no_norm.sh  "spring" "$seed" "data/spring_${NUM_TRAJECTORIES}.npy" "$NUM_TRAJECTORIES" "$EPOCHS" $HIDDEN_DIM 0.001 linear default linear linear prior_comparison_paper R_generic
        sbatch scripts/cluster_no_norm.sh  "spring" "$seed" "data/spring_${NUM_TRAJECTORIES}.npy" "$NUM_TRAJECTORIES" "$EPOCHS" $HIDDEN_DIM 0.001 linear quadratic linear linear prior_comparison_paper R_quadratic
        sbatch scripts/cluster_no_norm.sh  "spring" "$seed" "data/spring_${NUM_TRAJECTORIES}.npy" "$NUM_TRAJECTORIES" "$EPOCHS" $HIDDEN_DIM 0.001 default default mlp gradient_positive prior_comparison_paper default
    done
  done
done
 