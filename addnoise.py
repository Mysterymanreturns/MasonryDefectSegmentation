def addperlin1(image,variability):
    
    import numpy as np
    import random
    import noise
    
    shape = image.shape
    scale = 1
    octaves = 8
    persistence = 0.85
    lacunarity = 2.0
    seed = np.random.randint(0,100)

    world = np.zeros(shape)

    # make coordinate grid on [0,1]^2
    x_idx = np.linspace(0, 1, shape[0])
    y_idx = np.linspace(0, 1, shape[1])
    world_x, world_y = np.meshgrid(x_idx, y_idx)

    # apply perlin noise, instead of np.vectorize, consider using itertools.starmap()
    world = np.vectorize(noise.pnoise2)(world_x/scale,
                            world_y/scale,
                            octaves=octaves,
                            persistence=persistence,
                            lacunarity=lacunarity,
                            repeatx=1024,
                            repeaty=1024,
                            base=seed)

    # here was the error: one needs to normalize the image first. Could be done without copying the array, though
    img = world + .5
    imagerange = image.max()-image.min()
    imgrange = img.max()-img.min()
    img = img*variability*imagerange/imgrange
    
    image += img
    
    return image



def addperlin2(image,variability):
    
    import numpy as np
    import random
    import noise
    
    shape = image.shape
    scale = 1
    octaves = 2
    persistence = 0.2
    lacunarity = 2.0
    seed = np.random.randint(0,100)

    world = np.zeros(shape)

    # make coordinate grid on [0,1]^2
    x_idx = np.linspace(0, 1, shape[0])
    y_idx = np.linspace(0, 1, shape[1])
    world_x, world_y = np.meshgrid(x_idx, y_idx)

    # apply perlin noise, instead of np.vectorize, consider using itertools.starmap()
    world = np.vectorize(noise.pnoise2)(world_x/scale,
                            world_y/scale,
                            octaves=octaves,
                            persistence=persistence,
                            lacunarity=lacunarity,
                            repeatx=1024,
                            repeaty=1024,
                            base=seed)

    # here was the error: one needs to normalize the image first. Could be done without copying the array, though
    img = world + .5
    imagerange = image.max()-image.min()
    imgrange = img.max()-img.min()
    img = img*variability*imagerange/imgrange
    
    image += img
    
    return image

def addperlin3(image,variability):
    
    import numpy as np
    import random
    import noise
    
    shape = image.shape
    scale = 1
    octaves = 1
    persistence = 0.2
    lacunarity = 2.0
    seed = np.random.randint(0,100)

    world = np.zeros(shape)

    # make coordinate grid on [0,1]^2
    x_idx = np.linspace(0, 1, shape[0])
    y_idx = np.linspace(0, 1, shape[1])
    world_x, world_y = np.meshgrid(x_idx, y_idx)

    # apply perlin noise, instead of np.vectorize, consider using itertools.starmap()
    world = np.vectorize(noise.pnoise2)(world_x/scale,
                            world_y/scale,
                            octaves=octaves,
                            persistence=persistence,
                            lacunarity=lacunarity,
                            repeatx=1024,
                            repeaty=1024,
                            base=seed)

    # here was the error: one needs to normalize the image first. Could be done without copying the array, though
    img = world + .5
    imagerange = image.max()-image.min()
    imgrange = img.max()-img.min()
    img = img*variability*imagerange/imgrange
    
    image += img
    
    return image