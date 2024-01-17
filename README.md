# DatasetGeneration

This code repository includes the generation, loading, and training demos of three datasets: `cophy`, `vcdn` and `sprites`.

### Project structure

The project structure is as follows:

```
├── config
|   └── datagen_config.yaml // Dataset generation Settings
├── data					// Where the data set is stored
|   ├── balls				// From cophy, ball collision
|   ├── balls_2D			// From vcdn, ball collision
|   ├── blocktower			// From cophy, blocktower
|   ├── collision			// From cophy, a ball striking a cylinder
│   └── sprites				// Various characters moving around
├── data_generation				
|   ├── sprites_material	// Necessary material for sprite dataset
|   ├── urdf				// Necessary material for cophy dataset
|   ├── data_loader_xx.py
|   ├── generate_xx.py
|   └── utils.py
├── train_demo_cophy.py
├── train_demo_sprites.py
└── train_demo_vcdn.py
```

### Create the dataset

Please run the following command in the root directory. Here [scene] can be `balls`, `balls_2D`, `blocktower`, `collision`, and `sprites`

```
python ./data_generator/generate_[scene].py
```

This will create the dataset at `./data/[scene]`

If you want to change the generation settings, how many videos to generate or how many balls are there for example, please modify`./config/datagen_config.yaml`

### Run the demo

Please run the following command in the root directory.  Here [scene] can be `cophy`, `vcdn`, `sprites`

```
python ./train_demo_[source].py
```

This will show you the dimension and shape of the our data at the console.

To find out what each dimension of the data represents, please refer to the notes in demo files

