# CSCE-689-REINFORCEMENT-LEARNING-Project

## Quick Steps to run:
0. Navigate to Code Base
1. Install dependencies (cf requirements.txt)
2. Install donkeycar(Host PC) from [here](http://docs.donkeycar.com/guide/install_software/#step-1-install-software-on-host-pc) and follow the instruction to add a new car
3. Fill in the paths to the Supervised model and the car config file in sac2.py under algos
4. Train a control policy for 5000 steps using the modified Soft Actor-Critic (SAC)

```
python train.py --algo sac2 -vae2 path-to-vae.pkl -n 5000
```

5. Enjoy trained agent for 2000 steps
```
python enjoy.py --algo sac2 -vae path-to-vae.pkl --exp-id 0 -n 2000
```
6. Or try and already trained agent (model present under logs/sac2/DonkeyVae-v0-level-0_best/DonkeyVae-v0-level-0.pkl)
```
python enjoy.py --algo sac2 -vae path-to-vae.pkl --exp-id 0 -n 2000 --model logs/sac2/DonkeyVae-v0-level-0_best/DonkeyVae-v0-level-0.pkl
