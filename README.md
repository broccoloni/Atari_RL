# Atari_RL

This was for an assignment in a reinforcement learning class that I took.

This program needs ale-py and gym. After installing ale-py with pip, in the
command line run ale-import-roms roms/ to import the games in the roms 
directory.

The jupyter file visualize_envs contains the code to list and choose a game,
and select an algorithm to train an agent to play it. Note that as I haven't
implemented multiprocessing, the PG method is much quicker than the GA. All 
the functions I use to train the agents are in helper_functions.py, and the
network definition, from pytorch, are in AtariNet.py. Once the agents are 
trained, you can make a gif of them playing the game to see how they do.

An example of a gameplay gif when the agent is trained on 35000 games (1hr on slow computer) with PG
is below

<img src = "https://user-images.githubusercontent.com/38572823/225478735-827f2086-8e2a-4e37-b983-713e284e4d1e.gif" width = "250" height = "250" />
